from typing import Any, Optional

import ai2thor
import ai2thor.controller
import gym
import numpy as np
import torchvision.transforms as T
import torch
import torchvision.transforms.functional as F
import cv2

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import EnvType, SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super




class RawRGBSensorTHOR(Sensor):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        if isinstance(env, ai2thor.controller.Controller):
            return env.last_event.frame.copy()
        else:
            return env.current_frame.copy()


class RawDepthSensorTHOR(Sensor):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError


class RawNavigationStretchDepthSensor(RawDepthSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        depth_frame = env.navigation_depth_frame.copy()
        depth_frame = (
            np.clip(
                depth_frame,
                0.0,
                10.0,
            )
            * (255 / 10.0)
        ).astype(np.uint8)
        depth_frame = np.stack([depth_frame] * 3, axis=-1)

        return depth_frame


class RawManipulationStretchDepthSensor(RawDepthSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        depth_frame = env.manipulation_depth_frame.copy()
        depth_frame = (
            np.clip(
                depth_frame,
                0.0,
                10.0,
            )
            * (255 / 10.0)
        ).astype(np.uint8)
        depth_frame = np.stack([depth_frame] * 3, axis=-1)

        return depth_frame





class RawNavigationStretchRGBSensor(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        return env.navigation_camera.copy()


class RawManipulationStretchRGBSensor(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        return env.manipulation_camera.copy()
    





class CameraFollowAgent(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        return env.get_camera_behind_object_view().copy()


class TopDownPathViewRGBSensor(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        target_ids = None  # task.task_info.get("synset_to_object_ids", None)
        return env.get_top_down_path_view(task.task_info["followed_path"], target_ids).copy()
