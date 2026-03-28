import copy
import random
from abc import abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Type, Any

import numpy as np
from shapely import Point

from allenact.utils.system import get_logger
from tasks.abstract_task import AbstractVIDATask
from tasks.abstract_task_sampler import AbstractVIDATaskSampler
from utils.constants.stretch_initialization_utils import HORIZON
from utils.data_generation_utils.exception_utils import TaskSamplerException
from utils.data_generation_utils.navigation_utils import filtered_starting_positions
from utils.string_utils import strings_exist_in_dict_or_list
from utils.type_utils import Vector3, AgentPose, REGISTERED_TASK_PARAMS, AbstractTaskArgs
from utils.constants.object_constants import bad_asset_ids


class BaseTaskSampler(AbstractVIDATaskSampler):
    task_type_str: Optional[str] = None
    TELEPORT_THEN_SAMPLE: bool = True
    """
    Needs to be unique for each task type, and identical to the class name specifying
    the task type specific params (in utils.type_utils' REGISTERED_TASK_PARAMS)
    """

    def __init__(
        self,
        task_args: AbstractTaskArgs,
        houses: List[Dict],
        house_inds: List[int],
        target_object_types: List[str],
        controller_args: Dict,
        controller_type: Type,
        max_tasks: Union[int, float],
        sample_per_house: int,
        prob_randomize_materials: float = 0,
        task_type: Type = AbstractVIDATask,
        device: Type = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task_args=task_args,
            houses=houses,
            house_inds=house_inds,
            controller_args=controller_args,
            controller_type=controller_type,
            max_tasks=max_tasks,
            sample_per_house=sample_per_house,
            prob_randomize_materials=prob_randomize_materials,
            task_type=task_type,
            device=device,
            **kwargs,
        )
        # random.shuffle(self.house_inds) TODO Do not uncomment for data generation

        # get the total number of tasks assigned to this process
        self.max_tasks = max_tasks
        self.sample_per_house = sample_per_house

        self.target_object_types = target_object_types
        self.valid_rotations = np.arange(start=0, stop=360, step=30).tolist()

        self.reset()

        # Rose: This is unassigned and i'm not quite sure what it's supposed
        # to be - leaving alone for now. Multitaks sampler doesn't have a sampler-level
        # task type, in any case

        # assert (
        #     self.task_type_str in REGISTERED_TASK_PARAMS
        # ), f"{self.task_type_str} unknown in utils.type_utils.REGISTERED_TASK_PARAMS"

    @abstractmethod
    def sample_task_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def length(self) -> Union[int, float]:
        """Length.
        # Returns
        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return self._current_tasks_left

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self._current_tasks_left

    @property
    def last_sampled_task(self) -> Optional[AbstractVIDATask]:
        # NOTE: This book-keeping should be done in TaskSampler...
        return self._last_sampled_task

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.
        # Returns
        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    @property
    def current_house_index(self) -> int:
        return self.house_inds[self.house_iterator_index]

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.current_house_index]

    def increment_task_and_reset_house(
        self, force_advance_scene: bool, house_index: Optional[int] = None
    ):
        if house_index is not None:
            assert self.max_tasks == float("inf")
            self.house_iterator_index = self.house_inds.index(house_index)
            self.samples_per_current_house = 1
        else:
            self.increment_task(force_advance_scene=force_advance_scene)

        # The above code ensure that self.current_house will now be the next house
        self.reset_controller_in_current_house_and_cache_house_data(retain_agent_pose=False)

    def select_and_teleport_to_start(self, room_id_to_fsps, max_tries=1, params=None):
        # Returns starting pose if successful and if the task should be resampled (latter only used in subclasses)
        for _ in range(max_tries):
            starting_pose = AgentPose(
                position=random.choice(
                    room_id_to_fsps[random.choice(list(room_id_to_fsps.keys()))]
                ),
                rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
            )
            event = self.controller.teleport_agent(**starting_pose)

            if not event:
                get_logger().warning(
                    f"Teleport failing in {self.current_house_index} at {starting_pose}"
                )
                # get_logger().warning(event)
            else:
                return starting_pose, False
        return None, False

    def increment_task(self, force_advance_scene: bool):
        if (not force_advance_scene) and self.samples_per_current_house < self.sample_per_house:
            self.samples_per_current_house += 1
        else:
            self.house_iterator_index = (self.house_iterator_index + 1) % len(
                self.house_index_to_house
            )
            self.samples_per_current_house = 1

    def sample_valid_task_params(self, num_tries):
        if self.house_index_to_house[7] is None:  # this is such a hack TT_TT
            all_forbidden_object_ids = bad_asset_ids()["val"]
        else:
            all_forbidden_object_ids = bad_asset_ids()["train"]

        for i in range(num_tries):
            sampled_parameters = self.sample_task_parameters()
            forbidden_object_ids = (
                all_forbidden_object_ids[self.current_house_index]
                if self.current_house_index < len(all_forbidden_object_ids)
                else None
            )
            if not strings_exist_in_dict_or_list(sampled_parameters, forbidden_object_ids):
                return sampled_parameters
            print(
                f"Sampled bad asset in house {self.current_house_index} from {forbidden_object_ids} - resampling"
            )
        raise TaskSamplerException(
            f"Couldn't sample approved asset ids in house {self.current_house_index} after {num_tries} tries"
        )

    def next_task(
        self,
        force_advance_scene: bool = False,
        house_index: Optional[int] = None,
    ) -> Optional[AbstractVIDATask]:
        # NOTE: Stopping condition
        if self._current_tasks_left <= 0:
            return None

        self.increment_task_and_reset_house(
            force_advance_scene=force_advance_scene, house_index=house_index
        )
        assert house_index is None or self.current_house_index == house_index

        # Figure out which reachable positions are in each room so that we can first randomly
        # sample a room and then randomly sample a spawn location within that room (makes things more uniform)
        fsps = filtered_starting_positions(self.reachable_positions)
        room_poly_map = self.controller.room_poly_map
        room_id_to_fsps = defaultdict(lambda: [])
        for fsp in fsps:
            for room_id, poly in room_poly_map.items():
                if poly.contains(Point((fsp["x"], fsp["z"]))):
                    room_id_to_fsps[room_id].append(fsp)
                    break

        starting_pose, resample_params = None, False

        if self.TELEPORT_THEN_SAMPLE:
            starting_pose, resample_params = self.select_and_teleport_to_start(
                room_id_to_fsps, params=None
            )
            if starting_pose is None:
                raise TaskSamplerException("Teleportation failed")

        # Try teleporting to up to 5 positions (possibly near objects of interest) before rejecting the house
        for resample_try in range(5):
            if resample_try == 0 or resample_params:
                # sampled_parameters = self.sample_valid_task_params(num_tries=5) # TODO: revisit bad asset rejection in sampling
                sampled_parameters = self.sample_task_parameters()
                sampled_parameters["extras"] = {}

            if self.TELEPORT_THEN_SAMPLE:
                break

            starting_pose, resample_params = self.select_and_teleport_to_start(
                room_id_to_fsps, params=sampled_parameters
            )
            if starting_pose is not None:
                break

        if starting_pose is None:
            raise TaskSamplerException("Teleportation failed")

        self._current_tasks_left -= 1

        for param in REGISTERED_TASK_PARAMS[self.task_type_str]:
            assert (
                param in sampled_parameters
            ), f"Missing {param} from {self.task_type_str} (generated {list(sampled_parameters.keys())})"

        sampled_parameters.update(
            {
                "task_type": self.task_type_str,
                "num_rooms": len(self.house_index_to_house[self.current_house_index]["rooms"]),
                "house_index": str(self.current_house_index),
                "starting_pose": starting_pose,
            }
        )

        task_kwargs = dict(
            controller=self.controller,
            task_sampler=self,
            task_info=sampled_parameters,
        )
        self._last_sampled_task = self.task_type(
            **task_kwargs,
            **self.task_args,
            house=copy.deepcopy(self.house_index_to_house[self.current_house_index]),
        )
        return self._last_sampled_task

    def reset(self):
        self.house_iterator_index = -1  # -1 so that it doesn't skip the first task
        self.samples_per_current_house = 0
        self._current_tasks_left = self.max_tasks
        self.object_synset_counter = Counter()
