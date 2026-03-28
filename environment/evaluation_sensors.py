from allenact.utils.misc_utils import prepare_locals_for_super

from environment.stretch_controller import StretchController
from environment.stretch_state import StretchState
from tasks.abstract_task import AbstractVIDATask
import numpy as np
import gym
from allenact.base_abstractions.sensor import Sensor


class TargetbjectWasPickedUp(Sensor):
    def __init__(self, uuid: str = "target_obj_was_pickedup") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        target_obj_in_hand = False
        if "synsets" in task.task_info:
            object_types = task.task_info["synsets"]
            object_ids = []
            for object_type in object_types:
                object_ids += task.task_info["synset_to_object_ids"][object_type]
            objects_in_hand = StretchState(env).held_oids  # env.get_held_objects()
            target_obj_in_hand = len([x for x in objects_in_hand if x in object_ids]) > 0
        return np.array([target_obj_in_hand], dtype=np.int64)
