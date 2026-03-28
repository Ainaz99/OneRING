from dataclasses import dataclass, field
from typing import List

import fire
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOValue
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    TrainingSettings,
    LinearDecay,
)
from torch import optim
from architecture.models.spoc_models import REGISTERED_MODELS
from training.online.allenact_trainer import OnPolicyRunnerMixin
from training.online.siglip import SigLIPParams, SigLIP
from training.online.customized_loss import PPOLogGrad
from training.offline.train_utils import load_pl_ckpt, load_allenact_ckpt

def requires_grad(param):
    return param.requires_grad

@dataclass
class SigLIPTuneParams(SigLIPParams):
    # Lower the learning rate for finetuning
    lr: float = 2e-5
    num_steps: List[int] = field(default_factory=lambda: [64, 64, 64])
    batch_steps: List[int] = field(
        default_factory=lambda: [int(200000), int(10e6), int(1e9) - int(10e6) - int(200000)]
    )
    loss_names: List[str] = field(
        default_factory=lambda: ["ppo_value_loss", "ppo_loss", "ppo_loss"]
    )

    model: str = "TuneSpocLlamaModelWTextGoal"

    # IL checkpoint
    IL_train_id: str = ""
    IL_train_step: str = ""
    IL_basedir: str = "/tmp/il_checkpoints"

    # RL checkpoint
    RL_train_id: str = ""
    RL_train_step: str = ""
    RL_basedir: str = "/tmp/rl_checkpoints"

    # overwrite the BaseConfigParams tag
    tag: str = "ObjectNavType-RL-SigLIP-Tune-ViTb-TSFM"


class SigLIPTune(SigLIP):

    def __init__(
        self,
        params: "SigLIPTuneParams" = SigLIPTuneParams(),
    ):
        # Overwrite the default params to enable attribute references
        super().__init__(params)
        self.params = params
        self.model_pkg = REGISTERED_MODELS[self.params.model]

    def create_model(self, **kwargs) -> nn.Module:
        model = super().create_model(**kwargs)

        # Load the IL checkpoint
        if not self.params.resume_allenact_ckpt:
            if self.params.load_RL_ckpt:
                RL_ckpt = self.download_ckpt(
                    training_run_id=self.params.RL_train_id,
                    ckpt=self.params.RL_train_step,
                    output_basedir=self.params.RL_basedir,
                ) 
                load_pl_ckpt(model, RL_ckpt, ckpt_prefix="", verbose=True)
            else:
                IL_ckpt = self.download_ckpt(
                    training_run_id=self.params.IL_train_id,
                    ckpt=self.params.IL_train_step,
                    output_basedir=self.params.IL_basedir,
                )
                if "pl-model" in IL_ckpt:
                    print("--- Load Actor Model ---")
                    load_pl_ckpt(model, IL_ckpt, verbose=True)
                    print("--- Load Critic Model ---")
                    load_pl_ckpt(model.critic_spoc_model, IL_ckpt, verbose=True)
                else:
                    load_allenact_ckpt(model, IL_ckpt)

        # we only want objectnav actions
        non_nav_action_inds = [
            i
            for i, a in enumerate(self.params.actions.get_action_list())
            if a
            not in [
                action.to_string()
                for action in self.params.actions.base_actions
                + [self.params.actions.special_actions[0]]
            ]
        ]

        for i in non_nav_action_inds:
            model.actor.linear.bias.data[i] = -999999

        # Freeze action_decoder
        # for param in model.action_decoder.parameters():
        #     param.requires_grad = False
        # for param in model.critic_spoc_model.action_decoder.parameters():
        #     param.requires_grad = False

        # # Freeze actor_critic_heads
        # for param in model.actor.parameters():
        #     param.requires_grad = False

        # for param in model.critic.parameters():
        #     param.requires_grad = False

        # for param in model.critic_spoc_model.actor.parameters():
        #     param.requires_grad = False

        # for param in model.critic_spoc_model.critic.parameters():
        #     param.requires_grad = False

        # self.finetuned_params = list(filter(requires_grad, model.parameters()))

        return model

    def training_pipeline(self, **kwargs):
        assert len(self.params.num_steps) == len(self.params.batch_steps)
        assert len(self.params.num_steps) == len(self.params.loss_names)

        log_interval = [
            (
                self.params.num_train_processes * self.params.num_steps[idx] * 5
                if torch.cuda.is_available()
                else 1
            )
            for idx in range(len(self.params.num_steps))
        ]

        PPOConfigTune = dict(
            clip_param=0.1,
            value_loss_coef=0.5,
            entropy_coef=0.00,
            use_clipped_value_loss=False,
            action_loss_schedule=None,
            discrete_critics=False,
            normalize_advantage=False,
        )

        assert (
            self.params.advance_scene_rollout_period is None
        ), "use STEPS_IN_HOUSE_BEFORE_FORCE_SCENE_ADVANCE instead"

        

        return TrainingPipeline(
            save_interval=self.params.save_interval,
            metric_accumulate_interval=self.params.metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=self.params.lr)), # params=self.finetuned_params, 
            num_mini_batch=1,
            update_repeats=4,
            max_grad_norm=0.5,
            named_losses={
                "ppo_loss": PPOLogGrad(**PPOConfigTune),
                "ppo_value_loss": PPOValue(clip_param=0.1, use_clipped_value_loss=False),
            },
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            pipeline_stages=[
                PipelineStage(
                    loss_names=[self.params.loss_names[idx]],
                    max_stage_steps=self.params.batch_steps[idx],
                    training_settings=TrainingSettings(
                        num_steps=self.params.num_steps[idx],
                        metric_accumulate_interval=log_interval[idx],
                        advance_scene_rollout_period=self.params.steps_in_house_before_force_scene_advance
                        // self.params.num_steps[idx],
                        gae_lambda=0.95,
                    ),
                )
                for idx in range(len(self.params.num_steps))
            ],
        )



@dataclass
class SigLIPTuneRunner(SigLIPTuneParams, OnPolicyRunnerMixin):
    def get_config(self):
        return SigLIPTune(self)


if __name__ == "__main__":
    fire.Fire(SigLIPTuneRunner)
