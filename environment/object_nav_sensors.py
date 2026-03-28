import colorsys
import copy
import json
import math
import random
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, Tuple

import gym
import numpy as np
import torch
from allenact.base_abstractions.sensor import Sensor, SubTaskType
from allenact.base_abstractions.task import EnvType
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from environment.agent_parameter_utils import AgentParams


from environment.stretch_controller import StretchController
from environment.stretch_state import StretchState
from utils.constants.FPIN_utils import DEFAULT_ASSET
from utils.constants.stretch_initialization_utils import (
    AGENT_MODE,
)



from utils.sensor_constant_utils import LIST_OF_CAMERAS, get_len_agent_parameter_sensor, get_uuids_for_agent_parameter_sensor
from utils.string_utils import (
    convert_string_to_byte,
    json_templated_task_string,
)
from utils.task_spec_to_instruction import best_lemma
from utils.type_utils import get_task_relevant_synsets


if TYPE_CHECKING:
    from tasks.abstract_task import AbstractVIDATask
else:
    from typing import TypeVar

    AbstractVIDATask = TypeVar("AbstractVIDATask")
    
   
class AgentParametersSensor(Sensor):
    def __init__(
        self,
        uuid: str = "agent_parameter_sensor",
        str_max_len: Union[str, int] = "adaptive",
        datagen: bool = False,
    ) -> None:
        assert isinstance(str_max_len, int) or str_max_len == "adaptive"
        self.str_max_len = str_max_len
        self.datagen = datagen
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        if self.datagen:
            if self.str_max_len == "adaptive":
                return gym.spaces.MultiDiscrete([256] * 1)
            else:
                return gym.spaces.MultiDiscrete([256] * self.str_max_len)
            
        return gym.spaces.Discrete(get_len_agent_parameter_sensor(get_uuids_for_agent_parameter_sensor()))
    

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        # if sensor is used for data generation, we return the agent parameters as a string converted to bytes
        if self.datagen:
            agent_param_str = json.dumps(env.get_agent_parameters().to_dict())
            if self.str_max_len == "adaptive":
                bytes = convert_string_to_byte(agent_param_str, 2 * len(agent_param_str))
                final_index = len(bytes) + 1
                for ind in reversed(range(len(bytes))):
                    if bytes[ind] == 0:
                        final_index = ind
                return bytes[:final_index]
            elif isinstance(self.str_max_len, int):
                return convert_string_to_byte(agent_param_str, self.str_max_len)
            else:
                raise NotImplementedError
        else:
            # if the sensor is used for RL/evaluation, we return the agent parameters as a list of floats
            agent_param_dict = env.get_agent_parameters().to_dict()
            agent_parameter = []
            for i, uuid in enumerate(get_uuids_for_agent_parameter_sensor()):
                if 'rotation' in uuid:
                    # if we have the rotation in the format {'x': rot_x, 'y': 0, 'z': 0}
                    param = [agent_param_dict[uuid]['x'], agent_param_dict[uuid]['y']] 
                elif 'position' in uuid:
                    #{'x': pos_x, 'y': pos_y, 'z': pos_z}
                    param = list(agent_param_dict[uuid].values())
                elif 'body_params' in uuid:
                    param = [agent_param_dict[uuid]['originOffsetX'], agent_param_dict[uuid]['originOffsetZ']]
                    # param += list(agent_param_dict[uuid]['colliderScaleRatio'].values())
                    absolute_body_sizes = AgentParams.get_body_size_after_load(DEFAULT_ASSET)
                    collider_scales = agent_param_dict[uuid]['colliderScaleRatio']
                    scaled_body_sizes = [size * scale for size, scale in zip(absolute_body_sizes.values(), collider_scales.values())]
                    param += scaled_body_sizes

                elif isinstance(agent_param_dict[uuid], int):
                    param = [agent_param_dict[uuid]]
                else:
                    raise NotImplementedError(f"Agent parameter sensor {uuid} not implemented")
                agent_parameter += param
            return np.array(agent_parameter)
        
         
class AgentMaskSensor(Sensor):
    def __init__(self, uuid: str, height: int=480, width: int=640):
        self.height = height
        self.width = width
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.height, self.width), dtype=np.uint8
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        return task.task_info['agent_masks'][self.uuid]
       
class LastActionSuccessSensor(Sensor):
    def __init__(self, uuid: str = "last_action_success") -> None:
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
        return np.array([task.last_action_success], dtype=np.int64)


class LastActionIsRandomSensor(Sensor):
    def __init__(self, uuid: str = "last_action_is_random") -> None:
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
        return np.array([task.last_action_random], dtype=np.int64)


class LastAgentLocationSensor(Sensor):
    def __init__(self, uuid: str = "last_agent_location") -> None:
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
        agent_position_rotation = env.get_current_agent_full_pose()
        agent_position = agent_position_rotation["position"]
        agent_rotation = agent_position_rotation["rotation"]

        return np.array(
            [
                agent_position["x"],
                agent_position["y"],
                agent_position["z"],
                agent_rotation["x"],
                agent_rotation["y"],
                agent_rotation["z"],
            ],
            dtype=np.float64,
        )


class TaskTemplatedTextSpecSensor(Sensor):
    def __init__(
        self,
        uuid: str = "templated_task_spec",
        str_max_len: Union[str, int] = "adaptive",
        is_constant_across_episode: bool = True,
    ) -> None:
        assert isinstance(str_max_len, int) or str_max_len == "adaptive"
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        self.is_constant_across_episode = is_constant_across_episode
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.MultiDiscrete:
        if self.str_max_len == "adaptive":
            return gym.spaces.MultiDiscrete([256] * 1)
        else:
            return gym.spaces.MultiDiscrete([256] * self.str_max_len)

    @staticmethod
    def encode_observation(task_info, str_max_len: Union[str, int]):
        task_string = json_templated_task_string(task_info)
        if str_max_len == "adaptive":
            bytes = convert_string_to_byte(task_string, 2 * len(task_string))
            final_index = len(bytes) + 1
            for ind in reversed(range(len(bytes))):
                if bytes[ind] == 0:
                    final_index = ind
            return bytes[:final_index]
        elif isinstance(str_max_len, int):
            return convert_string_to_byte(task_string, str_max_len)
        else:
            raise NotImplementedError

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return self.encode_observation(task.task_info, self.str_max_len)


def round_floats_in_dict(target_dict, decimal_digits=4):
    if target_dict is None or type(target_dict) != dict:
        return target_dict
    target_dict = copy.deepcopy(target_dict)
    result_dict = {}
    for key, value in target_dict.items():
        if type(value) == float:
            formatted_float = round(value, decimal_digits)
            result_dict[key] = formatted_float
        elif type(value) == dict:
            result_dict[key] = round_floats_in_dict(value, decimal_digits)
        else:
            result_dict[key] = value
    return result_dict






class TaskNaturalLanguageSpecSensor(Sensor):
    def __init__(
        self,
        uuid: str = "task_natural_language_spec",
        str_max_len=1000,
    ) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> np.ndarray:

        natural_language_spec = task.task_info["natural_language_spec"]
        return convert_string_to_byte(natural_language_spec, self.str_max_len)


class HypotheticalTaskSuccessSensor(Sensor):
    def __init__(self, uuid: str = "hypothetical_task_success") -> None:
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
        return np.array([task.successful_if_done(strict_success=True)], dtype=np.int64)


class MinimumTargetAlignmentSensor(Sensor):
    def __init__(self, uuid: str = "minimum_visible_target_alignment") -> None:
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
        if "synsets" not in task.task_info:
            return np.array([-1], dtype=np.float64)
        object_type = task.task_info["synsets"][0]
        visible_target_objects = [
            obj
            for obj in task.task_info["synset_to_object_ids"][object_type]
            if task.controller.object_is_visible_in_camera(
                obj, which_camera="front", maximum_distance=2
            )
        ]
        visible_target_alignments = [
            abs(task.controller.get_agent_alignment_to_object(obj))
            for obj in visible_target_objects
        ]
        if len(visible_target_alignments) == 0:
            return np.array([-1], dtype=np.float64)
        else:
            return np.array([min(visible_target_alignments)], dtype=np.float64)


class Visible4mTargetCountSensor(Sensor):
    def __init__(self, uuid: str = "visible_target_4m_count") -> None:
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
        if "synsets" not in task.task_info:
            return np.array([0], dtype=np.float64)
        object_type = task.task_info["synsets"][0]
        visible_target_objects = [
            obj
            for obj in task.task_info["synset_to_object_ids"][object_type]
            if task.controller.object_is_visible_in_camera(
                obj, which_camera="front", maximum_distance=4
            )
        ]
        return np.array([len(visible_target_objects)], dtype=np.int64)


class MinL2TargetDistanceSensor(Sensor):
    def __init__(self, uuid: str = "minimum_l2_target_distance") -> None:
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
        if not hasattr(task, "min_l2_distance_to_target"):
            return np.array([-1], dtype=np.float64)
        return np.array([task.min_l2_distance_to_target()], dtype=np.float64)


class TargetObjectTypeStrSensor(Sensor):
    def __init__(
        self, uuid: str = "target_object_type", str_max_len=200, src="target_object_type"
    ) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        self.src = src
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(task.task_info[self.src], self.str_max_len)


class TargetObjectTypesStrSensor(Sensor):
    def __init__(self, uuid: str = "target_object_types", str_max_len=200, src="synsets") -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        self.src = src
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte("; ".join(task.task_info[self.src]), self.str_max_len)


class CurrentTargetObjectTypeStrSensor(Sensor):
    def __init__(
        self, uuid: str = "target_object_type", str_max_len=200, src="target_object_type"
    ) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        self.src = src
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(
            ", ".join(task.sub_task.task_info[self.src]), self.str_max_len  # type:ignore
        )


class TargetObjectIdStrSensor(Sensor):
    def __init__(self, uuid: str = "target_object_id", str_max_len=200) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(task.task_info["target_object_id"], self.str_max_len)


class TargetReceptacleTypeStrSensor(Sensor):
    def __init__(self, uuid: str = "target_receptacle_type", str_max_len=200) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(task.task_info["target_receptacle_type"], self.str_max_len)


class TargetRoomTypeStrSensor(Sensor):
    def __init__(self, uuid: str = "target_room_type", str_max_len=200) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(task.task_info["target_room_type"], self.str_max_len)


class SourceRoomTypeStrSensor(Sensor):
    def __init__(self, uuid: str = "source_room_type", str_max_len=200) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(task.task_info["source_room_type"], self.str_max_len)


class LastActionStrSensor(Sensor):
    def __init__(self, uuid: str = "last_action_str", str_max_len=200) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(task.last_taken_action_str, self.str_max_len)


class LastGoalAgentReferenceSensor(Sensor):
    def __init__(
        self, uuid: str = "last_goal_agent_frame", str_max_len=500, datagen: bool = False
    ) -> None:
        self.str_max_len = str_max_len
        self.datagen = datagen
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        relative_goal = round_floats_in_dict(task.agent_relative_goal)
        if self.datagen:
            return convert_string_to_byte(relative_goal, self.str_max_len)
        else:
            return np.array(get_quantized_goal(relative_goal))


class LastGoalAbsoluteReferenceSensor(Sensor):
    def __init__(self, uuid: str = "last_goal_absolute_frame", str_max_len=500) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        absolute_goal = round_floats_in_dict(task.absolute_goal)
        return convert_string_to_byte(absolute_goal, self.str_max_len)


class LastGoalHybridReferenceSensor(Sensor):
    def __init__(self, uuid: str = "last_goal_hybrid_frame", str_max_len=500) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if task.agent_relative_goal is None or "base_position" not in task.agent_relative_goal:
            return convert_string_to_byte(
                round_floats_in_dict(task.agent_relative_goal), self.str_max_len
            )
        else:
            try:
                hybrid_goal = json.loads(task.absolute_goal)
                hybrid_goal["base_position"] = json.loads(task.agent_relative_goal)["base_position"]
                return convert_string_to_byte(round_floats_in_dict(hybrid_goal), self.str_max_len)
            except:
                print("Error in LastGoalHybridReferenceSensor")


class LastActionTimeEstimateSensor(Sensor):
    def __init__(self, uuid: str = "last_action_time_estimate") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(task.last_action_time_estimate, dtype=np.float64)


class HouseNumberSensor(Sensor):
    def __init__(self, uuid: str = "house_index") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(int(task.task_info["house_index"]))


class GoalObjectTypeSensor(GoalObjectTypeThorSensor):
    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNaviThorGridTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        assert len(task.task_info["synsets"]) == 1
        return self.object_type_to_ind[task.task_info["synsets"][0]]


class TaskIdSensor(Sensor):
    def __init__(
        self,
        uuid: str = "task_id_sensor",
        **kwargs: Any,
    ):
        super().__init__(uuid=uuid, observation_space=gym.spaces.Discrete(1))

    def _get_observation_space(self):
        if self.target_to_detector_map is None:
            return gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            return gym.spaces.Discrete(len(self.detector_types))

    def get_observation(
        self,
        env,
        task,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "id" in task.task_info:
            id = task.task_info["id"]
        else:
            id = "no_task_id_provided"
        out = [ord(k) for k in id]
        for _ in range(len(out), 1000):
            out.append(ord(" "))
        return out


class RoomsSeenSensor(Sensor):
    def __init__(self, uuid: str = "rooms_seen") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(int(len(task.visited_and_left_rooms)))



        


class RepeatedFirstFrameNavhRGBSensorEval(Sensor):
    def __init__(self, uuid: str = "first_nav_frame_repeated") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:

        if task.num_steps_taken() == 0:
            return env.navigation_camera.copy()
        else:
            return task.observation_history[0][self.uuid].copy()



class IsObjectVisible(Sensor):
    def __init__(
        self, uuid: str = "is_obj_visible", which_camera: Literal["nav", "manip"] = "front"
    ) -> None:
        observation_space = self._get_observation_space()
        uuid = f"{uuid}_{which_camera}"
        super().__init__(**prepare_locals_for_super(locals()))
        self.which_camera = which_camera

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(
        self, env: StretchController, task: AbstractVIDATask, *args: Any, **kwargs: Any
    ) -> Any:
        if "object_id" in task.task_info:
            oid = task.task_info["object_id"]
            return int(
                env.object_is_visible_in_camera(
                    oid, which_camera=self.which_camera, maximum_distance=20
                )
            )
        else:
            return int(0)


class FinalDistanceToGoal(Sensor):
    def __init__(self, uuid: str = "final_distance") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(
        self, env: StretchController, task: AbstractVIDATask, *args: Any, **kwargs: Any
    ) -> float:
        goal = get_goal_3d_from_task(task)
        if goal is None:
            return 1000
        current_agent_location = env.get_current_agent_position()
        current_agent_location = np.array(
            [current_agent_location["x"], current_agent_location["z"]]
        )
        goal = np.array([goal["x"], goal["z"]])
        return np.linalg.norm(current_agent_location - goal)


class MinimumDistanceToGoal(Sensor):
    def __init__(self, uuid: str = "minimum_distance") -> None:
        observation_space = self._get_observation_space()
        self.current_minimum_distance = 1000
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(
        self, env: StretchController, task: AbstractVIDATask, *args: Any, **kwargs: Any
    ) -> float:
        goal = get_goal_3d_from_task(task)
        if goal is None:
            return self.current_minimum_distance
        if task.num_steps_taken() == 0:
            self.current_minimum_distance = 1000
        current_agent_location = env.get_current_agent_position()
        current_agent_location = np.array(
            [current_agent_location["x"], current_agent_location["z"]]
        )
        goal = np.array([goal["x"], goal["z"]])
        self.current_minimum_distance = min(
            self.current_minimum_distance, np.linalg.norm(current_agent_location - goal)
        )
        return self.current_minimum_distance


def get_goal_3d_from_task(task: AbstractVIDATask) -> Dict[str, float]:
    if "goal_in_world_3d" in task.task_info:
        return task.task_info["goal_in_world_3d"]
    if "target_obj_in_3d" in task.task_info:
        return task.task_info["target_obj_in_3d"]
    return None



class OriginalDistanceToGoal(Sensor):
    def __init__(self, uuid: str = "original_distance") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(
        self, env: StretchController, task: AbstractVIDATask, *args: Any, **kwargs: Any
    ) -> float:
        goal = get_goal_3d_from_task(task)
        if goal is None:
            return 1000

        if task.num_steps_taken() == 0:
            original_location = env.get_current_agent_position()
            original_location = np.array(
                [original_location["x"], original_location["y"], original_location["z"]]
            )
        else:
            original_location = task.observation_history[0]["last_agent_location"][:3]
        original_location = original_location[[0, 2]]
        goal = np.array([goal["x"], goal["z"]])
        return np.linalg.norm(original_location - goal)



class RoomCurrentSeenSensor(Sensor):
    def __init__(self, uuid: str = "room_current_seen") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(bool(task.get_current_room() in task.visited_and_left_rooms))


class CurrentAgentRoom(Sensor):
    def __init__(self, uuid: str = "current_agent_room") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        try:
            room_num = int(task.get_current_room().replace("room|", ""))
        except:
            room_num = -1
        return np.array([room_num], dtype=np.int64)


class NumPixelsVisible(Sensor):
    def __init__(
        self,
        which_camera: Literal["nav", "manip", "front", "left", "right", "down"],
        uuid: str = "num_pixels_visible",
    ) -> None:
        uuid = f"{uuid}_{which_camera}"
        self.which_camera = which_camera
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        num_pixels = 0

        if "synsets" in task.task_info and len(task.task_info["synsets"]) == 1:
            object_type = task.task_info["synsets"][0]
            object_ids = task.task_info["synset_to_object_ids"][object_type]
            num_pixels = 0

            visible_oids = env.get_visible_objects(
                which_camera=self.which_camera, maximum_distance=15
            )
            for oid in object_ids:
                if oid in visible_oids:
                    num_pixels += np.sum(
                        env.get_segmentation_mask_of_object(
                            object_id=oid, which_camera=self.which_camera
                        )
                    )

        return np.array([num_pixels], dtype=np.int64)



class TimeStepSensor(Sensor):
    def __init__(self, uuid: str = "time_ids", max_time_for_random_shift=0) -> None:
        observation_space = self._get_observation_space()
        self.max_time_for_random_shift = max_time_for_random_shift
        self.random_start = 0
        super().__init__(**prepare_locals_for_super(locals()))
        self._update = False

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def sample_random_start(self):
        self.random_start = random.randint(0, max(self.max_time_for_random_shift, 0))

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        steps = task.num_steps_taken()
        if self._update:
            steps += 1
        else:
            self._update = True
        if task.is_done():  # not increment at next episode start
            self._update = False
            self.sample_random_start()
        return np.array(self.random_start + int(steps), dtype=np.int64)


class TrajectorySensor(Sensor):
    def __init__(self, uuid: str = "traj_index", max_idx: int = 4) -> None:
        observation_space = self._get_observation_space()
        self.curr_idx = 0
        self.max_idx = max_idx
        self._update = False
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractVIDATask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if self._update:
            self.curr_idx += 1
            if self.curr_idx >= self.max_idx:
                self.curr_idx = 0
            self._update = False
        if task.is_done():  # update at next episode start
            self._update = True
        return np.array(self.curr_idx, dtype=np.int64)
