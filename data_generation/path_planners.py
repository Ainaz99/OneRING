import abc
import copy
import random
from collections import defaultdict
from typing import Sequence, Dict, Optional, Any, List, TYPE_CHECKING

from matplotlib import use
import networkx as nx
import numpy as np
from shapely import Point

from allenact.utils.system import get_logger
from environment.action_spaces import DONE_ACTION
from tasks.abstract_task import AbstractVIDATask
from tasks.object_nav_task import ObjectNavTask
from utils.data_generation_utils.exception_utils import (
    PlannerFailedToNavigateException,
    PlannerFailedToSeeException,
    TaskDifficultyIncorrectException,

)

from utils.data_generation_utils.navigation_utils import (
    visit_ids_from_info,
    walk_on_path,
    rotate_until_visible,
)
from utils.distance_calculation_utils import sum_dist_path
from utils.type_utils import REGISTERED_TASK_PARAMS
from data_generation.discrete_planner import DiscretePlanner



if TYPE_CHECKING:
    from environment.stretch_controller import StretchController

REGISTERED_PLANNERS: Dict[str, type] = {}


def register_planner(cls):
    # ignore planners without task_type_str
    if cls.task_type_str is None:
        return cls

    for task_str in cls.task_type_str:
        # ignore task_type_str not registered in REGISTERED_TASK_PARAMS
        if task_str not in REGISTERED_TASK_PARAMS:
            continue
        REGISTERED_PLANNERS[task_str] = cls
    return cls


class PathPlanner(abc.ABC):
    task_type_str: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_planner(cls)

    @abc.abstractmethod
    def plan(self, task: AbstractVIDATask) -> Optional[List[Dict[str, Any]]]:
        raise NotImplementedError

    @abc.abstractmethod
    def is_planner_guaranteed_to_fail(self, task: AbstractVIDATask) -> bool:
        return False



class ObjectNavPlanner(PathPlanner):
    task_type_str = [
        "ObjectNavType",
        "EasyObjectNavType",
        "HardObjectNavType",
        "ObjectNavRoom",
        "ObjectNavRelAttribute",
        "ObjectNavLocalRef",
        "ObjectNavAffordance",
        "ObjectNavOpenVocab",
    ]

    def __init__(
        self,
    ):
        pass

    def reset(self, task: ObjectNavTask):
        self.task = task
        self.controller = task.controller
        self.observations = []

    # TODO: is this superceded by the refactor?
    def is_planner_guaranteed_to_fail(self, task: AbstractVIDATask) -> bool:
        return False

    def plan(self, task: ObjectNavTask):
        self.reset(task)

        use_astar = True

        target_synset = task.task_info["synsets"][0]

        if use_astar:
            dp = DiscretePlanner(task)
            replan_kwargs = dict(
                replan_func=lambda task, *args, **kwargs: dp.blacklist_or_replan_to_loc(
                    *args, **kwargs
                )
            )
        else:
            dp = None
            replan_kwargs = {}

        if "visit_ids" in task.task_info:
            closest_object_id = visit_ids_from_info(task, maybe_dp=dp, replan_kwargs=replan_kwargs)
        else:
            closest_object_id = task.controller.get_closest_object_from_ids(
                task.task_info["synset_to_object_ids"][target_synset]
            )
        if closest_object_id is None:
            raise PlannerFailedToNavigateException(
                "Failed to visit alternatives or finding closest target"
            )

        if closest_object_id is None:
            raise PlannerFailedToNavigateException(
                f"Could not find shortest path to {closest_object_id} (object type {target_synset})."
            )

        if use_astar:
            _, path, _ = dp.nearest_object_id_path_dist([closest_object_id])
        else:
            path = task.controller.get_shortest_path_to_object(closest_object_id)

        if path is None:
            raise PlannerFailedToNavigateException(
                f"Could not find shortest path to {closest_object_id} (object type {target_synset})."
            )

        dist_to_object = sum_dist_path(path)
        if task.task_type_str == "EasyObjectNavType":
            if not 1 <= dist_to_object < 3:
                raise TaskDifficultyIncorrectException(
                    "For EasyObjectNavType, the distance should be between 1 and 3 meters."
                )
        elif task.task_type_str == "ObjectNavType":
            if dist_to_object <= 3:
                raise TaskDifficultyIncorrectException(
                    "For ObjectNavType, the distance should be greater than 3 meters."
                )
        elif task.task_type_str == "HardObjectNavType":
            if dist_to_object <= 10:
                raise TaskDifficultyIncorrectException(
                    "For HardObjectNavType, the distance should be greater than 10 meters."
                )

        task.add_extra_task_information("chosen_object_id", closest_object_id)

        nav_succ = walk_on_path(path, task, max_tries=task.max_steps, **replan_kwargs)

        if nav_succ:
            # Adding expert-selected object ID to task info
            task.add_extra_task_information("natural_language_description", task.to_string())

            # check if object is visible and issue done
            was_visible = rotate_until_visible(
                task, object_id=closest_object_id, use_task_success=True
            )

        else:
            raise PlannerFailedToNavigateException(
                f"Could not walk along shortest path to {closest_object_id} (object type {target_synset})."
            )

        task.step_with_random(task.get_action_from_goal(DONE_ACTION))

        if not task.is_successful():
            raise PlannerFailedToSeeException(f"Task unsuccessful after issuing done.")

        # Get observations from the task and return them
        if task.successful_if_done(strict_success=True):
            # print(task.to_string())
            # save_videos_hacky(task.get_observation_history(), note="success", burn_future_path=True)
            return task.get_observation_history()
        else:
            # save_videos_hacky(task.get_observation_history(), note="fail", burn_future_path=True)

            raise PlannerFailedToSeeException(
                f"Could not see {closest_object_id} (object type {target_synset}) after walking to it."
            )


