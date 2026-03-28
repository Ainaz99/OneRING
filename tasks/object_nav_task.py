from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING

from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.robothor_plugin.robothor_tasks import spl_metric
from environment.action_spaces import AbstractActionSpace
from tasks.abstract_task import AbstractVIDATask
from utils.data_generation_utils.navigation_utils import (
    is_any_object_sufficiently_visible_and_in_center_frame,
)


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

from utils.type_utils import RewardConfig

if TYPE_CHECKING:
    from environment.stretch_controller import StretchController

from training.online.reward_shaper import ObjectNavRewardShaper


class ObjectNavTaskSuccessMixIn:
    task_info: Optional[Dict[str, Any]] = None
    controller: Optional["StretchController"] = None

    def successful_if_done(self, strict_success=False) -> bool:
        object_type = self.task_info["synsets"][0]
        visible_targets = [
            oid
            for oid in self.task_info["broad_synset_to_object_ids"][object_type]
            if self.controller.object_is_visible_in_camera(
                oid, which_camera="nav", maximum_distance=2
            )
        ]
        if not strict_success:
            return len(visible_targets) > 0
        elif len(visible_targets) == 0:
            return False

        return is_any_object_sufficiently_visible_and_in_center_frame(
            controller=self.controller, object_ids=visible_targets, object_synset=object_type
        )


class ObjectNavTrackingMixIn:
    task_info: Optional[Dict[str, Any]] = None
    controller: Optional["StretchController"] = None
    _took_end_action: Optional[bool] = None
    reward_config: Optional[RewardConfig] = None
    dist_to_target_func: Optional[Callable] = None
    distance_type: Optional[str] = None
    closest_distance: Optional[float] = None
    _success: Optional[bool] = None
    max_steps: Optional[int] = None
    _rewards: Optional[List[float]] = None
    _reward_shaper: Optional[ObjectNavRewardShaper] = None

    @property
    def reward_shaper(self) -> ObjectNavRewardShaper:
        if self.reward_config is not None:
            if self._reward_shaper is None:
                self._reward_shaper = ObjectNavRewardShaper(task=self)
            return self._reward_shaper
        else:
            return None

    def min_l2_distance_to_target(self) -> float:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = float("inf")

        target_object_ids = sum(
            map(list, self.task_info["broad_synset_to_object_ids"].values()), []
        )
        for object_id in target_object_ids:
            min_dist = min(
                min_dist,
                IThorEnvironment.position_dist(
                    self.controller.get_obj_pos_from_obj_id(object_id),
                    self.controller.get_current_agent_position(),
                ),
            )
        if min_dist == float("inf"):
            get_logger().error(
                f"No target object among {target_object_ids} found"
                f" in house {self.task_info['house_index']}."
            )
            return -1.0
        return min_dist

    def min_geodesic_distance_to_target(self) -> float:
        target_object_ids = sum(
            map(list, self.task_info["broad_synset_to_object_ids"].values()), []
        )
        _, min_dist = self.controller.get_closest_object_from_ids(
            object_ids=target_object_ids, return_id_and_dist=True
        )
        return min_dist

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        if self.reward_config.shaping_weight == 0.0:
            return 0
        return self.reward_shaper.shaping()

    def judge(self) -> float:
        """Judge the last event."""
        if self.reward_config is None:
            return 0
        reward = self.reward_config.step_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        return float(reward)


class ObjectNavTask(ObjectNavTaskSuccessMixIn, ObjectNavTrackingMixIn, AbstractVIDATask):
    task_type_str = "ObjectNavType"

    def __init__(
        self,
        controller: "StretchController",
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        action_space: AbstractActionSpace,
        reward_config: Optional[RewardConfig] = None,
        distance_type: Literal["l2"] = "l2",
        visualize: Optional[bool] = None,
        house: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self.path: List = []  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0
        self.last_taken_action_str = ""
        self.last_action_success = -1
        self.last_action_random = -1

        self.distance_type = distance_type
        self.dist_to_target_func = self.min_l2_distance_to_target

        if (
            self.task_info.get("synset_to_object_ids") is None
        ):  # Kiana : added this line, ask before removing
            # if "synsets" not in self.task_info:
            #     #  REMOVE THIS ONCE THE NEW DATASET IS GENERATED
            #     self.task_info = TARGET_TO_CONVERTER["ToLatest"].convert_params(
            #         self.task_info["task_type"], self.task_info, verbose=False
            #     )
            # TODO: Can this be removed now? 23JAN2024
            self.task_info["synset_to_object_ids"] = {
                synset: [
                    o["objectId"]
                    for o in self.controller.get_all_objects_of_synset(
                        synset=synset, include_hyponyms=True
                    )
                ]
                for synset in self.task_info["synsets"]
            }

        last_distance = self.dist_to_target_func()
        self.closest_distance = last_distance
        self.optimal_distance = (
            last_distance
            if self.dist_to_target_func == self.min_geodesic_distance_to_target
            else self.min_geodesic_distance_to_target()  # TODO: What's going on here?
        )

        self.visualize = visualize

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        metrics["ep_length"] = self.num_steps_taken()
        metrics["dist_to_target"] = self.dist_to_target_func()
        metrics["total_reward"] = np.sum(self._rewards)
        metrics["spl"] = spl_metric(
            success=self._success,
            optimal_distance=self.optimal_distance,
            travelled_distance=self.travelled_distance,
        )
        metrics["spl"] = (
            0.0 if metrics["spl"] is None or np.isnan(metrics["spl"]) else metrics["spl"]
        )
        metrics["success"] = self._success

        self._metrics = metrics

        return metrics


class EasyObjectNavTask(ObjectNavTask):
    task_type_str = "EasyObjectNavType"


class HardObjectNavTask(ObjectNavTask):
    task_type_str = "HardObjectNavType"


class ObjectNavRoomTask(ObjectNavTask):
    task_type_str = "ObjectNavRoom"
    # add a room id to task_info


class ObjectNavRelAttributeTask(ObjectNavTask):  # TODO Maybe move to MultiON?
    task_type_str = "ObjectNavRelAttribute"
    # add a room id to task_info


class ObjectNavLocalRefTask(ObjectNavTask):  # TODO Maybe move to MultiON?
    task_type_str = "ObjectNavLocalRef"
    # add a room id to task_info


class ObjectNavAffordanceTask(ObjectNavTask):
    task_type_str = "ObjectNavAffordance"


class ObjectNavOpenVocabTask(ObjectNavTask):
    task_type_str = "ObjectNavOpenVocab"
