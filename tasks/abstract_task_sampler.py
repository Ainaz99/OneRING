import abc
import gc
import random
import sys
import warnings
from typing import Dict, List, Optional, Union, Type, Any, TYPE_CHECKING

import ai2thor.platform
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import set_seed
from torch.distributions.utils import lazy_property

from tasks.abstract_task import AbstractVIDATask
from utils.constants.FPIN_utils import DEFAULT_ASSET
from utils.constants.stretch_initialization_utils import PHYSICS_SETTLING_TIME
from utils.data_generation_utils.exception_utils import (
    HouseInvalidForTaskException,
    TaskSamplerInInvalidStateError,
)
from utils.type_utils import Vector3, AbstractTaskArgs, KeyedDefaultDict

if TYPE_CHECKING:
    from environment.stretch_controller import StretchController

import ai2thor.platform


class AbstractVIDATaskSampler(TaskSampler):
    def __init__(
        self,
        task_args: AbstractTaskArgs,
        houses: List[Dict],
        house_inds: List[int],
        controller_args: Dict,
        controller_type: Type,
        prob_randomize_materials: float = 0,
        task_type: Optional[Type] = None,
        device: Optional[int] = None,
        controller: Optional["StretchController"] = None,
        always_allocate_a_new_stretch_controller_when_reset: bool = False,
        settle_physics_for_second_when_reset: float = PHYSICS_SETTLING_TIME,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.controller_type = controller_type

        self._given_controller = controller

        self.always_allocate_a_new_stretch_controller_when_reset = (
            always_allocate_a_new_stretch_controller_when_reset
        )
        self.settle_physics_for_seconds_when_reset = settle_physics_for_second_when_reset
        assert (
            PHYSICS_SETTLING_TIME == settle_physics_for_second_when_reset
        ), "Currently not allowed! Chat with Luca/Kiana before allowing different values."

        house_index_to_local_index = {
            house_index: local_index for local_index, house_index in enumerate(house_inds)
        }
        self.house_index_to_house = KeyedDefaultDict(
            lambda house_index: houses[house_index_to_local_index[house_index]]
        )
        self.house_inds = house_inds

        assert len(houses) == len(house_inds)
        self.prob_randomize_materials = prob_randomize_materials
        self.task_args = task_args

        self.controller_args = controller_args

        if device is not None and device != -1 and sys.platform != "darwin":
            self.controller_args = {
                **self.controller_args,
                "platform": ai2thor.platform.CloudRendering,
                "gpu_device": device,
            }

        assert self.controller_args["agentMode"] != "locobot"

        self._last_sampled_task: Optional[AbstractVIDATask] = None

        self.reachable_positions_map: Dict[int, List[Vector3]] = {}

        self.visible_objects_cache = {}

        self.fixed_starting_positions = {}

    def set_seed(self, seed: int):
        set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.
        # Returns
        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        raise NotImplementedError

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        raise NotImplementedError

    @property
    def last_sampled_task(self) -> Optional[AbstractVIDATask]:
        # NOTE: This book-keeping should be done in TaskSampler...
        raise NotImplementedError

    def close(self) -> None:
        if self._given_controller is None and self.controller is not None:
            self.controller.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.
        # Returns
        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def current_house_index(self) -> int:
        raise NotImplementedError

    @property
    def current_house(self) -> Dict:
        return self.house_index_to_house[self.current_house_index]

    @lazy_property
    def controller(self):
        if self._given_controller is None:
            try:
                return self.controller_type(**self.controller_args)
            except Exception as e:
                if "Unity process has exited" in e.args[0]:
                    raise TaskSamplerInInvalidStateError("Controller has closed.")
                else:
                    raise
        else:
            return self._given_controller

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.current_house_index]

    def reset_controller_in_current_house_and_cache_house_data(
        self, skip_controller_reset: bool = False, retain_agent_pose: bool = False
    ) -> None:
        """Prepare the house for sampling tasks."""
        if not skip_controller_reset:
            agent_pose = None
            if retain_agent_pose:
                agent_pose = self.controller.get_current_agent_full_pose()

            if self.always_allocate_a_new_stretch_controller_when_reset:
                self.allocate_a_new_stretch_controller(use_original_ai2thor_controller=True)

            if self.current_house is None:
                raise HouseInvalidForTaskException(
                    "Current house is None. This can happen if the house was not successfully generated."
                )

            try:
                if self.current_house_index in self.fixed_starting_positions:
                    self.current_house["metadata"]["agent"]["position"] = (
                        self.fixed_starting_positions[self.current_house_index]
                    )

                self.reset_scene_with_timeout_handler()
            except ValueError as e:
                if "write to closed file" in e.args[0]:
                    raise TaskSamplerInInvalidStateError("Controller has closed.")
                else:
                    raise

            if retain_agent_pose:
                self.controller.teleport_agent(
                    position=agent_pose["position"],
                    rotation=agent_pose["rotation"],
                    forceAction=True,
                )
            else:
                if (
                    self.controller.get_current_agent_full_pose()["position"]["y"] < 0
                    or len(self.controller.get_reachable_positions()) < 1
                ):
                    warnings.warn(
                        f"Initial teleport failed in {self.current_house_index}, attempting to fix this initial position."
                    )
                    # Teleportation failed, let's try to find some other position
                    teleport_success = False
                    for room_id in sorted(self.controller.room_poly_map.keys()):
                        candidate_points = self.controller.get_candidate_points_in_room(room_id)
                        print(f"candidate_points: {candidate_points}")
                        for cand in candidate_points:
                            print(f"cand: {cand}")
                            print(f"self.current_house['metadata']['agent']['position']: {self.current_house['metadata']['agent']['position']}")
                            event = self.controller.teleport_agent(
                                position={
                                    "x": float(cand[0]),
                                    "y": self.current_house["metadata"]["agent"]["position"]["y"],
                                    "z": float(cand[1]),
                                },
                                rotation=self.current_house["metadata"]["agent"]["rotation"],
                            )
                            if event and event.metadata.get("lastActionSuccess", False):
                                teleport_success = True
                                break

                        if teleport_success:
                            break

                    if teleport_success:
                        assert self.controller.get_current_agent_full_pose()["position"]["y"] > 0
                        self.fixed_starting_positions[self.current_house_index] = (
                            self.controller.get_current_agent_full_pose()["position"]
                        )
                    else:
                        raise HouseInvalidForTaskException(
                            "Could not find a valid teleportation point in the house."
                        )

            if self.settle_physics_for_seconds_when_reset > 0:
                self.controller.step(
                    action="AdvancePhysicsStep",
                    simSeconds=self.settle_physics_for_seconds_when_reset,
                    raise_for_failure=True,
                )

        # NOTE: Set reachable positions
        if self.current_house_index not in self.reachable_positions_map:
            self.reachable_positions_map[self.current_house_index] = (
                self.controller.get_reachable_positions()
            )

        self.randomize_materials()  # Must be done after resetting!

    def reset_scene_with_timeout_handler(self):
        # if "ceiling_objects" in self.current_house:
        #     cids = {o["id"] for o in self.current_house["ceiling_objects"]}
        #     objs_to_keep = [o for o in self.current_house["objects"] if o["id"] not in cids]
        #     self.current_house["objects"] = objs_to_keep
        #     self.current_house.pop("ceiling_objects")

        try:
            reset_agent_params = None
            if DEFAULT_ASSET == "ghost":
                # For evaluating on FPIN benchmark with random agents, we have to reset based on the agent parameter sensor
                reset_agent_params = self.current_task_spec.get('eval_info', {}).get('agent_parameter_sensor', None)
                # For training RL with random agents, we have to reset based on the agent parameter sensor
                if reset_agent_params is None:
                    reset_agent_params = self.current_task_spec.get('agent_parameter_sensor', None)

            self.controller.reset(scene=self.current_house, reset_agent_params=reset_agent_params)
        except:
            self.allocate_a_new_stretch_controller(use_original_ai2thor_controller=False)
            self.controller.reset(scene=self.current_house)

    def allocate_a_new_stretch_controller(self, use_original_ai2thor_controller):
        if use_original_ai2thor_controller:
            c = self.controller.controller
            self.controller.controller = None
        self.controller.stop = lambda *args, **kwargs: None
        del self.controller
        gc.collect()

        try:
            self.controller = self.controller_type(
                initialize_controller=not use_original_ai2thor_controller, **self.controller_args
            )
        except TimeoutError:
            if hasattr(self, "controller"):
                del self.controller
                gc.collect()
            self.controller = self.controller_type(**self.controller_args)

        if use_original_ai2thor_controller:
            self.controller.controller = c

    def randomize_materials(self):
        if random.random() < self.prob_randomize_materials:
            self.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.controller.step(action="ResetMaterials", raise_for_failure=True)

    def increment_task_and_reset_house(
        self, force_advance_scene: bool, house_index: Optional[int] = None
    ) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        raise NotImplementedError

    def next_task(
        self,
        force_advance_scene: bool = False,
        house_index: Optional[int] = None,
    ) -> Optional[AbstractVIDATask]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
