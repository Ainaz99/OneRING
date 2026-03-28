from typing import Literal, Union, Callable
from typing import Literal, Union, Callable
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn as nn
from architecture.models.spoc_models.common.preprocessors import (
    SigLipPreprocessorNoImagePaddingConfig,
)
from architecture.models.spoc_models.spoc_model import SpocModelConfig, SpocModel
from architecture.models.spoc_models.common.action_decoder import (
    LlamaActionDecoderConfig,
    LlamaActionDecoder,
    GoalCondLlamaActionDecoderConfig,
    GoalCondLlamaActionDecoder,
)
from utils.sensor_constant_utils import is_a_visual_sensor



@dataclass
class SpocLlamaModelConfig(SpocModelConfig):
    batch_size: int = 224
    max_seq_len: int = 320
    dropout: float = 0.1
    dropout: float = 0.1

    @property
    def action_decoder_config(self) -> LlamaActionDecoderConfig:
        return LlamaActionDecoderConfig(
            action_space=self.action_space,
            n_layers=3,
            dim=512,
            n_heads=8,
            output_size=512,
            # It is the batch size for the KV cache matrices in the llama decoder.
            # Optimize it for your GPU memory. It should equal to the batch size at training.
            max_batch_size=self.batch_size,
            # It is the maximum sequence length for the KV cache matrices in the llama decoder.
            # Optimize it for your GPU memory.
            # It should equal to the window size at training and max allowed steps at testing.
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
        )


class SpocLlamaModel(SpocModel):
    def __init__(self, cfg: SpocLlamaModelConfig, train_mode: Literal["IL", "RL"] = "IL"):
        super().__init__(cfg, train_mode)
        # Overwrite the cfg to avoid complains from type checker
        self.cfg = cfg
        self.curr_obs_frames = None
        
    @property
    def get_curr_obs_frames(self):
        return self.curr_obs_frames

    def build_action_decoder(self):
        return LlamaActionDecoder(self.cfg.action_decoder_config)

    def cache_visual_embeds(
        self,
        visual_embeds: torch.Tensor,
    ):
        # We don't need to cache the visual embeddings for Llama decoder
        # Instead, the Llama decoder does the KV-cache in itself
        pass

    def construct_causal_mask(
        self, observations: dict[str, torch.Tensor], is_single_time_step: bool
    ):
        if is_single_time_step:
            # We need causal mask for both RL and IL once it is single time step,
            # since we are doing online rollout now!
            time_steps = observations["time_ids"]  # B, T
            epi_start = torch.clamp(self.kv_cache_step_counter - time_steps, min=0).expand(
                -1, self.kv_cache_step_counter + 1
            )  # B, 1
            step_range = torch.arange(0, self.kv_cache_step_counter + 1).to(device=epi_start.device)
            causal_mask = torch.Tensor(epi_start <= step_range).unsqueeze(1).unsqueeze(1)
        else:
            if self.train_mode == "IL":
                # In IL mode, we don't need causal mask
                causal_mask = None
            elif self.train_mode == "RL":
                # In RL mode, we need causal mask
                traj_idx = observations["traj_index"]  # B, T
                causal_mask = torch.Tensor(traj_idx[:, :, None] == traj_idx[:, None, :])
                causal_mask = torch.tril(causal_mask).unsqueeze(1)
            else:
                raise ValueError(f"Unknown train_mode: {self.train_mode}")
        if causal_mask is not None:
            causal_mask = causal_mask.float()
            causal_mask[causal_mask == 0] = float("-inf")
            causal_mask[causal_mask == 1] = 0
        return causal_mask

    def forward_decoder(
        self,
        observations: dict[str, torch.Tensor],
        visual_embeds: torch.Tensor,
        adapted_text_features: torch.Tensor,
        prev_actions: torch.Tensor,
    ):
        # Use feature cache
        is_single_time_step = visual_embeds.shape[1] == 1
        if is_single_time_step and self.train_mode == "IL":
            time_ids = torch.Tensor([[self.kv_cache_step_counter]]).to(visual_embeds.device)
            observations["time_ids"] = time_ids
            prev_actions = prev_actions[:, -1].unsqueeze(1)
        else:
            time_ids = observations["time_ids"]

        # Construct causal mask
        causal_mask = self.construct_causal_mask(observations, is_single_time_step)

        # Decode the action conditioned on the goal conditioned visual features using a causal transformer
        decoder_output = self.action_decoder(
            goal_cond_visual_feats=visual_embeds,
            goals_features=adapted_text_features,
            last_actions=prev_actions,
            time_ids=time_ids,
            causal_mask=causal_mask,
            padding_mask=observations.get("padding_mask", None),
            an_object_is_in_hand=observations.get("an_object_is_in_hand", None),
            kv_cache_step_counter=self.kv_cache_step_counter,
        )
        if is_single_time_step:
            self.kv_cache_step_counter += 1
        else:
            self.kv_cache_step_counter = 0
        return decoder_output
    

    def get_action(self, observations: dict[str, Union[torch.Tensor, str]], goal: str):
        obs = {
            key: torch.tensor(value).unsqueeze(0)
            for (key, value) in observations.items()
            if isinstance(value, np.ndarray)
        }
        
        obs["goal"] = goal
        obs["last_actions"] = self.cache["last_actions"]
        preproc_obs = self.preproc.process([{"input_sensors": obs}])
        preproc_obs["time_ids"] = (
            torch.arange(preproc_obs["lengths"][0]).unsqueeze(0).to(self.preproc.device)
        )

        action_logits = self.forward(
            observations=preproc_obs, prev_actions=preproc_obs["last_actions"]
        )[0, -1]
        action_index = torch.distributions.categorical.Categorical(logits=action_logits).sample()
        action = self.cfg.action_space.get_action_list()[action_index.item()]

        self.cache["last_actions"].append(action)
        ############################################
        # Get the frames for visualization
        frames_dict = {
            sensor: frame
            for (sensor, frame) in preproc_obs.items()
            if is_a_visual_sensor(sensor)
        }
        reverse_normalized_frames = [
            (
                ((frame.squeeze().permute(1, 2, 0).cpu().numpy() * self.cfg.preproc_config.stdev) + self.cfg.preproc_config.mean) * 255
            ).astype(np.uint8)
            for frame in frames_dict.values()
        ]
        self.curr_obs_frames = np.concatenate(reverse_normalized_frames, axis=1)
        ############################################

        return action, torch.softmax(action_logits, -1)


@dataclass
class SpocGoalCondLlamaModelConfig(SpocModelConfig):
    batch_size: int = 224
    max_seq_len: int = 320
    dropout: float = 0.1
    use_rms_norm: bool = True
    norm_first: bool = True
    activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = nn.functional.silu

    @property
    def action_decoder_config(self) -> GoalCondLlamaActionDecoderConfig:
        return GoalCondLlamaActionDecoderConfig(
            action_space=self.action_space,
            n_layers=3,
            dim=512,
            n_heads=8,
            output_size=512,
            # It is the batch size for the KV cache matrices in the llama decoder.
            # Optimize it for your GPU memory. It should equal to the batch size at training.
            max_batch_size=self.batch_size,
            # It is the maximum sequence length for the KV cache matrices in the llama decoder.
            # Optimize it for your GPU memory.
            # It should equal to the window size at training and max allowed steps at testing.
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            use_rms_norm=self.use_rms_norm,
            norm_first=self.norm_first,
            activation=self.activation,
        )


class SpocGoalCondLlamaModel(SpocLlamaModel):
    def __init__(self, cfg: SpocGoalCondLlamaModelConfig, train_mode: Literal["IL", "RL"] = "IL"):
        super().__init__(cfg, train_mode)
        # Overwrite the cfg to avoid complains from type checker
        self.cfg = cfg

    def build_action_decoder(self):
        return GoalCondLlamaActionDecoder(self.cfg.action_decoder_config)


@dataclass
class SpocGoalCondLlamaModelNoImagePaddingConfig(SpocGoalCondLlamaModelConfig):
    @property
    def preproc_config(self):
        return SigLipPreprocessorNoImagePaddingConfig(action_space=self.action_space)
    

class SpocGoalCondLlamaModelSeparate(SpocGoalCondLlamaModel):
    def __init__(self, cfg: SpocGoalCondLlamaModelConfig, train_mode: Literal["IL", "RL"] = "IL"):
        super().__init__(cfg, train_mode)
        # Overwrite the cfg to avoid complains from type checker
        self.cfg = cfg

        self.critic_spoc_model = SpocGoalCondLlamaModel(cfg, train_mode)

    def clone_observation(self, observations: dict):
        if isinstance(observations, torch.Tensor):
            return observations.clone()

        o = {}
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                o[key] = value.clone()
            elif isinstance(value, dict):
                o[key] = self.clone_observation(value)
            else:
                raise ValueError(f"Unknown observation type: {type(value)}")
        return o

    def forward(self, *args, **kwargs):
        if self.train_mode == "IL":
            return super().forward(*args, **kwargs)
        elif self.train_mode == "RL":
            observations_clone = self.clone_observation(kwargs["observations"])
            prev_actions_clone = self.clone_observation(kwargs["prev_actions"])
            actor_output, memory = super().forward(*args, **kwargs)
            critic_output, critic_memory = self.critic_spoc_model(
                *args,
                observations=observations_clone,
                prev_actions=prev_actions_clone,
                memory=kwargs["memory"],
                masks=kwargs["masks"],
            )

            critic_output.distributions = actor_output.distributions
            
            return critic_output, critic_memory
        else:
            raise ValueError(f"Unknown train_mode: {self.train_mode}")


class SpocLlamaModelSeparate(SpocLlamaModel):
    def __init__(self, cfg: SpocLlamaModelConfig, train_mode: Literal["IL", "RL"] = "IL"):
        super().__init__(cfg, train_mode)
        # Overwrite the cfg to avoid complains from type checker
        self.cfg = cfg

        self.critic_spoc_model = SpocLlamaModel(cfg, train_mode)

    def clone_observation(self, observations: dict):
        if isinstance(observations, torch.Tensor):
            return observations.clone()

        o = {}
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                o[key] = value.clone()
            elif isinstance(value, dict):
                o[key] = self.clone_observation(value)
            else:
                raise ValueError(f"Unknown observation type: {type(value)}")
        return o

    def forward(self, *args, **kwargs):
        if self.train_mode == "IL":
            return super().forward(*args, **kwargs)
        elif self.train_mode == "RL":
            observations_clone = self.clone_observation(kwargs["observations"])
            prev_actions_clone = self.clone_observation(kwargs["prev_actions"])
            actor_output, memory = super().forward(*args, **kwargs)
            critic_output, critic_memory = self.critic_spoc_model(
                *args,
                observations=observations_clone,
                prev_actions=prev_actions_clone,
                memory=kwargs["memory"],
                masks=kwargs["masks"],
            )

            critic_output.distributions = actor_output.distributions
            return critic_output, critic_memory
        else:
            raise ValueError(f"Unknown train_mode: {self.train_mode}")



if __name__ == "__main__":
    cfg = SpocLlamaModelConfig()
    model = SpocLlamaModel(cfg)
    print(cfg)
