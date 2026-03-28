import os
import random
from typing import Optional, List, Literal, Tuple, Type, Union, Any
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gym

from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.base_abstractions.distributions import CategoricalDistr
from environment.action_spaces import SPOCV1ActionSpace
from architecture.models.spoc_models.common.preprocessors import (
    SigLipPreprocessorConfig,
    SigLipPreprocessor,
    RLImageAugmentationPreprocessor,
    RLImageAugmentationPreprocessorConfig,
    RLTextPreprocessor,
    RLTextPreprocessorConfig,
)
from architecture.models.spoc_models.common.goal_cond_visual_encoder import (
    GoalCondVisualEncoder,
    GoalCondVisualEncoderConfig,
)
from architecture.models.spoc_models.common.action_decoder import (
    ActionDecoder,
    ActionDecoderConfig,
)
from architecture.models.spoc_models.common.utils import LinearActorHead, LinearCriticHead
from training.offline.train_utils import load_pl_ckpt, load_allenact_ckpt
from utils.sensor_constant_utils import (
    is_a_goal_features,
    is_a_visual_features,
    is_a_visual_sensor,
)



@dataclass
class SpocModelConfig:
    # observations is a list of observation uuids that are used in the model
    observations: List[str] = None

    action_space: SPOCV1ActionSpace = None

    goal_text_padding_length: int = 0


    pass_agent_param_to_decoder: bool = False


    # image_encoder:Tuple[Type[nn.Module], SigLIPConfig] = IMAGE_ENCODER.SigLIPBase

    @property
    def preproc_config(self):
        return SigLipPreprocessorConfig(
            action_space=self.action_space, goal_text_padding_length=self.goal_text_padding_length
        )

    @property
    def frames_features_dim(self):
        # frames_features_dim is used in GoalCondVisualEncoder
        # It should be determined by which image encoder is used
        # For example, SigLIP's frames_features_dim is 768, while
        # DINOv2's frames_features_dim is 384
        return self.preproc_config.image_encoder.value[1].frames_features_dim

    @property
    def goal_text_features_dim(self):
        # goal_text_features_dim is used in GoalCondVisualEncoder
        # It should be determined by which text encoder is used
        # For example, SigLIP's goal_text_features_dim is 768, while
        # T5's goal_text_features_dim is 512
        return self.preproc_config.text_encoder.value[1].goal_text_features_dim

    @property
    def visual_features(self) -> List[str]:
        return [obs for obs in self.observations if is_a_visual_features(obs)]

    @property
    def goals_features(self) -> List[str]:
        return [obs for obs in self.observations if is_a_goal_features(obs)]
    

    @property
    def goal_text_uuid(self) -> str:
        if "goal_text_features" in self.observations:
            return "goal_text_features"
        else:
            return None


    @property
    def visual_encoder_config(self) -> GoalCondVisualEncoderConfig:
        return GoalCondVisualEncoderConfig(
            visual_features=self.visual_features,
            goal_text_uuid=self.goal_text_uuid,
            frames_features_dim=self.frames_features_dim,
            goal_text_features_dim=self.goal_text_features_dim,
        )

    @property
    def action_decoder_config(self) -> ActionDecoderConfig:
        return ActionDecoderConfig(action_space=self.action_space)


class SpocModel(nn.Module):
    def __init__(self, cfg: SpocModelConfig, train_mode: Literal["IL", "RL"] = "IL"):
        super().__init__()
        self.cfg = cfg
        self.train_mode = train_mode
        self.visual_encoder = self.build_visual_encoder()
        self.action_decoder = self.build_action_decoder()
        self.actor, self.critic = self.build_actor_critic_heads()
        self.preproc: Optional[SigLipPreprocessor] = None


        self.kv_cache_step_counter = 0
        self.start_token = self.cfg.action_space.get_num_actions()
        self.cache = {}
        self.reset()
        self.curr_obs_frames = None
        self.height_preds = []
        self.prev_height = -1


    @property
    def get_curr_obs_frames(self):
        return self.curr_obs_frames

    @classmethod
    def build_agent(
        cls,
        cfg: SpocModelConfig,
        ckpt_pth: Optional[str] = None,
        device: str = "cpu",
        load_from_pl: bool = True,
    ):
        spoc_model = cls(cfg)
        if ckpt_pth is not None:
            if load_from_pl:
                load_pl_ckpt(spoc_model, ckpt_pth)
            else:
                load_allenact_ckpt(spoc_model, ckpt_pth)
        spoc_model = spoc_model.to(torch.device(device))
        spoc_model.preproc = spoc_model.build_preproc(cfg.preproc_config)
        spoc_model.preproc.to(torch.device(device))
        return spoc_model

    def reset(self):
        self.kv_cache_step_counter = 0
        self.cache = {
            "visual_embeds": [],
            "time_ids": None,
            "last_actions": [""],
        }

    def get_action_list(self):
        return self.cfg.action_space.get_action_list()

    @classmethod
    def build_preproc(cls, preproc_config, train_mode: Literal["IL", "RL"] = "IL"):
        # We don't want to put the entire preproc in the model, since we don't want to save Dino-v2 giant or T5 weights
        # However, it is welcome to construct the preproc using this function for training loop.
        if train_mode == "IL":
            return SigLipPreprocessor(preproc_config, device=torch.device("cpu"))
        elif train_mode == "RL":
            return [
                RLImageAugmentationPreprocessor(
                    RLImageAugmentationPreprocessorConfig(
                        rgb_input_uuid=preproc_config.rgb_input_uuid,
                        feats_uuid=preproc_config.feats_uuid,
                        output_uuid="visual_feats",
                        image_encoder=preproc_config.image_encoder,
                    )
                ),
                RLTextPreprocessor(
                    RLTextPreprocessorConfig(
                        text_uuid=preproc_config.text_input_uuid,
                        output_uuid=preproc_config.goal_text_uuid,
                        text_encoder=preproc_config.text_encoder,
                        goal_text_padding_length=preproc_config.goal_text_padding_length,
                    )
                ),
            ]
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def build_visual_encoder(self):
        return GoalCondVisualEncoder(self.cfg.visual_encoder_config)


    def build_action_decoder(self):
        return ActionDecoder(self.cfg.action_decoder_config)

    def build_actor_critic_heads(self) -> tuple[LinearActorHead, LinearCriticHead]:
        return LinearActorHead(
            self.action_decoder.cfg.decoder.dim, self.cfg.action_space.get_num_actions()
        ), LinearCriticHead(self.action_decoder.cfg.decoder.dim)



    @property
    def recurrent_memory_specification(self):
        return None

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.cfg.action_space.get_num_actions())

    def prepare_input_from_batch(
        self, batch: dict[str, torch.Tensor], prev_actions: torch.Tensor
    ) -> (dict[str, Any], torch.Tensor):
        if self.train_mode == "IL":
            # The batch is produced by the preprocessor from IL
            observations = {key: batch[key] for key in self.cfg.observations}
        elif self.train_mode == "RL":
            # The batch is produced by the preprocessor from RL
            batch.update(batch["visual_feats"])
            observations = {
                key: value.movedim(0, 1)
                for (key, value) in batch.items()
                if isinstance(value, torch.Tensor)
            }

            # As the prev_actions is set to 0 for the 1st time step in AllenAct, we need to turn it to the START_TOKEN
            prev_actions = prev_actions.movedim(0, 1)
            prev_actions[observations["time_ids"] == 0] = (
                self.cfg.action_space.get_action_index_from_string("")
            )
        else:
            raise ValueError(f"Unknown train_mode: {self.train_mode}")

        return observations, prev_actions

    def forward_encoder(self, observations: dict[str, torch.Tensor]):
        # Get the input features from the observations dictionary
        frames_features_dict = {
            key: value for (key, value) in observations.items() if key in self.cfg.visual_features
        }

        # Get the goal features from the observations dictionary
        goals_features_dict = {key: observations[key] for key in self.cfg.goals_features}
        
        
        # Encode goal conditioned visual features using a non-causal transformer
        visual_embeds, adapted_text_features = self.visual_encoder(
            frames_features_dict=frames_features_dict, goals_features_dict=goals_features_dict
        )

        return visual_embeds, adapted_text_features

    def cache_visual_embeds(
        self,
        visual_embeds: torch.Tensor,
    ):
        # Cache the visual embeddings and generate the time ids according to the length of the cache
        self.cache["visual_embeds"].append(visual_embeds)
        self.cache["time_ids"] = (
            torch.arange(len(self.cache["visual_embeds"])).to(visual_embeds.device).unsqueeze(0)
        )

    def construct_causal_mask(
        self, observations: dict[str, torch.Tensor], is_single_time_step: bool
    ):
        if self.train_mode == "IL":
            # In IL mode, we don't need causal mask
            causal_mask = None
        elif self.train_mode == "RL":
            raise NotImplementedError(f"The regular PyTorch Causal Decoder won't for the RL mode")
        else:
            raise ValueError(f"Unknown train_mode: {self.train_mode}")
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
        if is_single_time_step:
            # Cache the visual embeddings if the trajectory index is the same
            self.cache_visual_embeds(visual_embeds)
            visual_embeds = torch.cat(self.cache["visual_embeds"], dim=1)
            time_ids = self.cache["time_ids"]
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
        )
        return decoder_output

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        prev_actions: torch.Tensor,
        masks: Optional[torch.Tensor] = None,  # This is the GRU legacy used in AllenAct
        memory: Optional[Memory] = None,  # This is the GRU legacy used in AllenAct
    ):
        observations, prev_actions = self.prepare_input_from_batch(observations, prev_actions)

        # Encode the visual features and goal features using the goal conditioned visual encoder
        visual_embeds, adapted_text_features = self.forward_encoder(observations)

        # Encode the memory using the causal transformer decoder
        decoder_output = self.forward_decoder(
            observations, visual_embeds, adapted_text_features, prev_actions
        )

        # Get the logits for the actions
        action_logits = self.actor(decoder_output)

        extras = {}

        # Handle different output formats for IL and RL
        if self.train_mode == "IL":
            # If it is IL mode, return the logits directly
            return action_logits
        elif self.train_mode == "RL":
            # If it is RL mode, return the action distribution as well as the estimated value
            values = self.critic(decoder_output)
            action_logits = action_logits.movedim(0, 1)
            values = values.movedim(0, 1)
            
            return (
                ActorCriticOutput(
                    distributions=CategoricalDistr(logits=action_logits), values=values, extras=extras
                ),
                memory,
            )
        else:
            raise ValueError(f"Unknown train_mode: {self.train_mode}")

    def get_action(self, observations: dict[str, Union[torch.Tensor, str]], goal: str):
        obs = {
            key: torch.tensor(value).unsqueeze(0)
            for (key, value) in observations.items()
            if isinstance(value, np.ndarray)
        }
        obs["goal"] = goal
        obs["last_actions"] = self.cache["last_actions"]
        preproc_obs = self.preproc.process([{"input_sensors": obs}])

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


if __name__ == "__main__":
    cfg = SpocModelConfig()
    model = SpocModel(cfg)
    print(cfg)
