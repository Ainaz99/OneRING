from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Sequence, List

from environment.object_nav_sensors import TaskNaturalLanguageSpecSensor
import fire
import torch
import torch.nn as nn
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    TrainingSettings,
)
from torch import optim

from architecture.models.spoc_models import REGISTERED_MODELS
from architecture.models.spoc_models.common.preprocessors import RLUnifiedPreprocessorConfig
from architecture.models.spoc_models.common.image_encoder import IMAGE_ENCODER
from architecture.models.spoc_models.common.text_encoder import TEXT_ENCODER
from data_generation.task_datagen_utils import get_core_sensors
from training.online.allenact_trainer import OnPolicyRunnerMixin
from training.online.base import BaseConfig, BaseConfigParams
from utils.type_utils import RewardConfig
from utils.wandb_logging import SimpleWandbLogging
from utils.sensor_constant_utils import is_a_visual_sensor


def get_additional_sensors_for_RL():
    return [
        TaskNaturalLanguageSpecSensor(uuid="goal"),
    ]

@dataclass
class SigLIPParams(BaseConfigParams):
    use_data_augmentation: bool = True
    traj_max_index: int = 2048
    use_traj_indexing: bool = True
    advance_scene_rollout_period: Optional[int] = None
    steps_in_house_before_force_scene_advance: int = 2000
    wandb_project: str = "FPIN-rl"
    wandb_entity: str = "prior-ai2"
    collision_penalty: float = 0.0
    shaping_weight: float = 1.0
    lr: float = 2e-4
    num_steps: List[int] = field(default_factory=lambda: [32, 64, 128])
    batch_steps: List[int] = field(
        default_factory=lambda: [int(10e6), int(10e6), int(1e9) - int(10e6) - int(10e6)]
    )

    # preprocess params
    preproc_config: RLUnifiedPreprocessorConfig = None

    image_encoder: IMAGE_ENCODER = IMAGE_ENCODER.SigLIPBase
    text_encoder: TEXT_ENCODER = TEXT_ENCODER.SigLIPBase

    # training pipeline params
    save_interval: int = 2_500_000
    metric_accumulate_interval: int = 10_000

    # overwrite the BaseConfigParams tag
    tag: str = "ObjectNavType-RL-SigLIP-ViTb-TSFM"

    # sensors and feats uuid used in preproc
    # WE SHOULDN'T CHANGE IT BY AUGMENTS
    rgb_input_uuid: List[str] = field(default_factory=lambda: [])
    feats_uuid: List[str] = field(default_factory=lambda: [])
    text_uuid: str = ""


class SigLIP(BaseConfig):

    def __init__(
        self,
        params: SigLIPParams = SigLIPParams(),
    ):
        super().__init__(params)
        # Overwrite the default params to enable attribute references
        self.params = params
        self.model_pkg = REGISTERED_MODELS["SpocLlamaModelWTextGoal"]

    def make_sampler_fn(self, **kwargs):
        kwargs["task_args"]["reward_config"] = RewardConfig(
            step_penalty=-0.01,
            goal_success_reward=10.0,
            failed_stop_reward=0.0,
            shaping_weight=self.params.shaping_weight,
            reached_horizon_reward=0.0,
            positive_only_reward=False,
            failed_action_penalty=self.params.collision_penalty,
        )
        return super().make_sampler_fn(**kwargs)

    def preprocessors(self) -> Sequence[Preprocessor]:
        # Overwite preproc config
        self.params.preproc_config = RLUnifiedPreprocessorConfig(
            rgb_input_uuid=[
                sensor for sensor in self.model_pkg.input_sensors if is_a_visual_sensor(sensor)
            ],
            feats_uuid=self.model_pkg.config.visual_features,
            image_encoder=self.params.image_encoder,
            text_encoder=self.params.text_encoder,
            text_input_uuid="goal",
            goal_text_uuid=self.model_pkg.config.goal_text_uuid,
            goal_text_padding_length=self.model_pkg.config.goal_text_padding_length,
        )

        if len(self._preproc) == 0:
            self._preproc = self.model_pkg.model_cls.build_preproc(
                self.params.preproc_config, train_mode="RL"
            )
        return self._preproc

    @cached_property
    def sensors(self):
        try:
            h, w = self.model_pkg.config.preproc_config.image_size
        except:
            h, w = 256, 256
        # we use padded RGB sensors for navigation and manipulation instead of the original ones

        sensors = [
            sensor for sensor in get_core_sensors() if sensor.uuid in self.model_pkg.input_sensors 
        ]
        
        sensors += [
            sensor
            for sensor in get_additional_sensors_for_RL()
            if sensor.uuid in self.model_pkg.input_sensors
        ]

        
        
        return sensors

    def create_model(self, **kwargs) -> nn.Module:
        model = self.model_pkg.build_model(train_mode="RL")

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

        return model

    def training_pipeline(self, **kwargs):
        assert len(self.params.num_steps) == len(self.params.batch_steps)

        log_interval = [
            (
                self.params.num_train_processes * self.params.num_steps[idx] * 5
                if torch.cuda.is_available()
                else 1
            )
            for idx in range(len(self.params.num_steps))
        ]

        assert (
            self.params.advance_scene_rollout_period is None
        ), "use STEPS_IN_HOUSE_BEFORE_FORCE_SCENE_ADVANCE instead"

        return TrainingPipeline(
            save_interval=self.params.save_interval,
            metric_accumulate_interval=self.params.metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=self.params.lr)),
            num_mini_batch=1,
            update_repeats=4,
            max_grad_norm=0.5,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=self.params.batch_steps[idx],
                    training_settings=TrainingSettings(
                        num_steps=self.params.num_steps[idx],
                        metric_accumulate_interval=log_interval[idx],
                        advance_scene_rollout_period=self.params.steps_in_house_before_force_scene_advance
                        // self.params.num_steps[idx],
                    ),
                )
                for idx in range(len(self.params.num_steps))
            ],
        )

    def wandb_logging_callback(self) -> SimpleWandbLogging:
        assert self.params.wandb_entity is not None and self.params.wandb_project is not None, (
            "Entity and project must be set to use wandb logging."
            " Set these values when specifying the --config_kwargs when running the experiment."
        )
        return SimpleWandbLogging(
            project=self.params.wandb_project,
            entity=self.params.wandb_entity,
            output_dir=self.params.output_dir,
            extra_tag=self.params.tag,
        )


@dataclass
class SigLIPRunner(SigLIPParams, OnPolicyRunnerMixin):
    def get_config(self):
        return SigLIP(self)


if __name__ == "__main__":
    fire.Fire(SigLIPRunner)
