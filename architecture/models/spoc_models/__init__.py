from dataclasses import dataclass
from typing import List, Type, Any, Literal
import torch.nn as nn
from architecture.models.spoc_models.spoc_llama_model import (
    SpocLlamaModel,
    SpocLlamaModelConfig,
    SpocGoalCondLlamaModel,
    SpocGoalCondLlamaModelConfig,
    SpocGoalCondLlamaModelSeparate,
)
from environment.action_spaces import SPOCV1ActionSpace


REGISTERED_MODELS = {}


@dataclass
class SPOCModelPackage:
    model_cls: Type
    config: Any
    input_sensors: List[str]

    def build_model(self, train_mode: Literal["IL", "RL"] = "IL"):
        return self.model_cls(self.config, train_mode=train_mode)


REGISTERED_MODELS["SpocLlamaModelWTextGoal"] = SPOCModelPackage(
    model_cls=SpocGoalCondLlamaModel,
    config=SpocGoalCondLlamaModelConfig(
        observations=[
            "raw_navigation_camera_features",
            "raw_manipulation_camera_features",
            "goal_text_features",
            "time_ids",
            "last_actions",
            "traj_index",
            "padding_mask",
        ],
        action_space=SPOCV1ActionSpace(),
    ),
    input_sensors=[
        "raw_navigation_camera",
        "raw_manipulation_camera",
        "goal",
        "time_ids",
        "last_actions",
        "traj_index",
    ],
)

REGISTERED_MODELS["TuneSpocLlamaModelWTextGoal"] = SPOCModelPackage(
    model_cls=SpocGoalCondLlamaModelSeparate,
    config=REGISTERED_MODELS["SpocLlamaModelWTextGoal"].config,
    input_sensors=REGISTERED_MODELS["SpocLlamaModelWTextGoal"].input_sensors,
)