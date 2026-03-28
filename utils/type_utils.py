import sys
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Set

import numpy as np

from environment.action_spaces import AbstractActionSpace
from utils.objaverse_utils import _OBJAVERSE_ANNOTATIONS

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

from allenact.base_abstractions.sensor import Sensor
from attrs import define

_TRAIN_TAIL_CATEGORIES: Optional[Set[str]] = None


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class KeyedDefaultDict(defaultdict):
    """
    A defaultdict that passes the key to the default_factory.
    """

    def __missing__(self, key: Any):
        return self.default_factory(key)


@define
class TaskSamplerArgs:
    process_ind: int
    """The process index number."""

    mode: Literal["train", "eval"]
    """Whether we are in training or evaluation mode."""

    house_inds: List[int]
    """Which houses to use for each process."""

    houses: Any
    """The hugging face Dataset of all the houses in the split."""

    sensors: List[Sensor]
    """The sensors to use for each task."""

    controller_args: Dict[str, Any]
    """The arguments to pass to the AI2-THOR controller."""

    reward_config: Dict[str, Any]
    """The reward configuration to use."""

    target_object_types: List[str]
    """The object types to use as targets."""

    action_names: List[str]
    """The object types to use as targets."""

    max_steps: int
    """The maximum number of steps to run each task."""

    max_tasks: int
    """The maximum number of tasks to run."""

    distance_type: str
    """The type of distance computation to use ("l2" or "geo")."""

    resample_same_scene_freq: int
    """
    Number of times to sample a scene/house before moving to the next one.

    If <1 then will never 
        sample a new scene (unless `force_advance_scene=True` is passed to `next_task(...)`.
    ."""

    # Can we remove?
    deterministic_cudnn: bool = False
    loop_dataset: bool = True
    seed: Optional[int] = None
    allow_flipping: bool = False


@define
class RewardConfig:
    step_penalty: float
    goal_success_reward: float
    failed_stop_reward: float
    shaping_weight: float
    reached_horizon_reward: float
    positive_only_reward: bool
    failed_action_penalty: float = 0.0


class AgentPose(TypedDict):
    position: Vector3
    rotation: Vector3


class AbstractTaskArgs(TypedDict):
    sensors: List[Sensor]
    max_steps: int
    action_space: AbstractActionSpace
    reward_config: Optional[RewardConfig]


class ObjectNavTaskArgs(AbstractTaskArgs):
    pass


class ObjectExploreTaskArgs(AbstractTaskArgs):
    pass


class Fetch2RoomTaskArgs(AbstractTaskArgs):
    pass


class Fetch2SurfaceTaskArgs(AbstractTaskArgs):
    pass


class FetchTaskArgs(AbstractTaskArgs):
    pass


class THORActions:
    move_ahead = "m"
    move_back = "b"
    rotate_right = "r"
    rotate_left = "l"
    rotate_right_small = "rs"
    rotate_left_small = "ls"
    done = "end"
    move_arm_up = "yp"
    move_arm_up_small = "yps"
    move_arm_down = "ym"
    move_arm_down_small = "yms"
    move_arm_out = "zp"
    move_arm_out_small = "zps"
    move_arm_in = "zm"
    move_arm_in_small = "zms"
    wrist_open = "wp"  # TODO: THESE ARE BACKWARDS? DECREASES WRIST DEGREES
    wrist_close = "wm"  # TODO: THESE ARE BACKWARDS? INCREASES WRIST DEGREES
    pickup = "p"
    dropoff = "d"
    ARM_ACTIONS = [
        move_arm_in,
        move_arm_out,
        move_arm_up,
        move_arm_down,
        move_arm_in_small,
        move_arm_out_small,
        move_arm_up_small,
        move_arm_down_small,
    ]
    MOVE_ACTIONS = [
        move_ahead,
        move_back,
    ]
    ROTATE_ACTIONS = [
        rotate_right,
        rotate_left,
        rotate_right_small,
        rotate_left_small,
    ]
    sub_done = "sub_done"

    @classmethod
    def get_action_name(cls, short_string):
        for name, value in cls.__dict__.items():
            if value == short_string:
                return name
        return None


REGISTERED_TASK_PARAMS: Dict[str, List[str]] = {}

if sys.version_info >= (3, 9):

    def get_required_keys(cls):
        return getattr(cls, "__required_keys__", [])

else:

    def get_required_keys(cls):
        return list(cls.__annotations__.keys())


def register_task_specific_params(cls):
    REGISTERED_TASK_PARAMS[cls.__name__] = get_required_keys(cls)
    return cls


class ObjectInstr(TypedDict):
    synsets: List[str]


class ObjectEval(TypedDict):
    synset_to_object_ids: Dict[str, List[str]]
    broad_synset_to_object_ids: Dict[str, List[str]]


class ObjectNav(ObjectInstr, ObjectEval):
    pass


class Fetch(ObjectInstr, ObjectEval):
    pass


class ObjRoom(TypedDict):
    room_type: str


class RequiresVisits(TypedDict):
    visit_ids: Dict[str, List[str]]


class RelAttribute(RequiresVisits, ObjRoom):
    rel_attribute: Union[str, Tuple[str, str]]


class LocalRef(RequiresVisits):
    reference_type: str
    reference_synsets: List[str]


class Affordance(TypedDict):
    affordance: str


class RoomInstr(TypedDict):
    dest_room_type: str


class RoomEval(TypedDict):
    dest_room_ids: List[str]


class ToRoom(RoomInstr, RoomEval):
    pass


class SurfaceInstr(TypedDict):
    dest_receptacle_synset: str


class SurfaceEval(TypedDict):
    dest_receptacle_ids: List[str]
    broad_dest_receptacle_ids: List[str]


class ToSurface(SurfaceInstr, SurfaceEval):
    pass


class RefInstr(TypedDict):
    dest_synset: str


class RefEval(TypedDict):
    dest_object_ids: List[str]
    broad_dest_object_ids: List[str]


class ToRef(RefInstr, RefEval):
    pass


class OpenVocab(TypedDict):
    uid: str


class NumRequired(TypedDict):
    num_required: int


@register_task_specific_params
class ObjectNavType(ObjectNav):
    pass


@register_task_specific_params
class EasyObjectNavType(ObjectNav):
    pass


@register_task_specific_params
class HardObjectNavType(ObjectNav):
    pass


@register_task_specific_params
class ExploreObjectNavType(ObjectNav):
    pass


@register_task_specific_params
class HardExploreObjectNavType(ObjectNav):
    pass


@register_task_specific_params
class ObjectNavRoom(ObjectNav, ObjRoom):
    pass


@register_task_specific_params
class ObjectNavRelAttribute(ObjectNav, RelAttribute):
    pass


@register_task_specific_params
class ObjectNavAffordance(ObjectNav, Affordance):
    pass


@register_task_specific_params
class ObjectNavLocalRef(ObjectNav, LocalRef):
    pass


@register_task_specific_params
class ObjectNavOpenVocab(ObjectNav, OpenVocab):
    pass


@register_task_specific_params
class ObjectNavMulti(ObjectNav):
    pass


@register_task_specific_params
class ObjectNavMofN(ObjectNav, NumRequired):
    pass


@register_task_specific_params
class ObjectNavRoomMulti(ObjectNav, ObjRoom, NumRequired):
    multi: int  # placeholder for the task to know what to remove from pending items when visiting a target


@register_task_specific_params
class FetchType(Fetch):
    pass


@register_task_specific_params
class EasyFetchType(Fetch):
    pass


@register_task_specific_params
class FetchObjRoom(Fetch, ObjRoom):
    pass


@register_task_specific_params
class FetchRelAttribute(Fetch, RelAttribute):
    pass


@register_task_specific_params
class FetchAffordance(Fetch, Affordance):
    pass


@register_task_specific_params
class FetchLocalRef(Fetch, LocalRef):
    pass


@register_task_specific_params
class FetchOpenVocab(Fetch, OpenVocab):
    pass


@register_task_specific_params
class PickupType(Fetch):
    pass


@register_task_specific_params
class PickupRelAttribute(Fetch, RelAttribute):
    pass


@register_task_specific_params
class PickupLocalRef(Fetch, LocalRef):
    pass


@register_task_specific_params
class PickupAffordance(Fetch, Affordance):
    pass


@register_task_specific_params
class PickupOpenVocab(Fetch, OpenVocab):
    pass


@register_task_specific_params
class Fetch2RoomType(Fetch, ToRoom):
    pass


@register_task_specific_params
class Fetch2RoomObjRoom(Fetch, ToRoom, ObjRoom):
    pass


@register_task_specific_params
class Fetch2RoomRelAttribute(Fetch, ToRoom, RelAttribute):
    pass


@register_task_specific_params
class Fetch2RoomAffordance(Fetch, ToRoom, Affordance):
    pass


@register_task_specific_params
class Fetch2RoomLocalRef(Fetch, ToRoom, LocalRef):
    pass


@register_task_specific_params
class Fetch2RoomOpenVocab(Fetch, ToRoom, OpenVocab):
    pass


@register_task_specific_params
class Fetch2RefType(Fetch, ToRef):
    pass


@register_task_specific_params
class Fetch2RefObjRoom(Fetch, ToRef, ObjRoom):
    pass


@register_task_specific_params
class Fetch2RefRelAttribute(Fetch, ToRef, RelAttribute):
    pass


@register_task_specific_params
class Fetch2RefAffordance(Fetch, ToRef, Affordance):
    pass


@register_task_specific_params
class Fetch2RefOpenVocab(Fetch, ToRef, OpenVocab):
    pass


@register_task_specific_params
class Fetch2SurfaceType(Fetch, ToSurface):
    pass


@register_task_specific_params
class Fetch2SurfaceObjRoom(Fetch, ToSurface, ObjRoom):
    pass


@register_task_specific_params
class Fetch2SurfaceRelAttribute(Fetch, ToSurface, RelAttribute):
    pass


@register_task_specific_params
class Fetch2SurfaceAffordance(Fetch, ToSurface, Affordance):
    pass


@register_task_specific_params
class Fetch2SurfaceLocalRef(Fetch, ToSurface, LocalRef):
    pass


@register_task_specific_params
class Fetch2SurfaceOpenVocab(Fetch, ToSurface, OpenVocab):
    pass


@register_task_specific_params
class Fetch2SurfaceMultiObject(Fetch, ToSurface):
    pass


@register_task_specific_params
class RoomNav(TypedDict):
    room_types: List[str]
    room_ids: Dict[str, List[str]]


@register_task_specific_params
class ExploreHouse(TypedDict):
    pass


@register_task_specific_params
class SimpleExploreHouse(TypedDict):
    pass


@register_task_specific_params
class GoToPointOnFloor(TypedDict):
    goal_in_camera_2d_first_step: Tuple[float, float]
    goal_in_world_3d: Dict[str, float]
    which_camera: Literal["nav", "manip"]


@register_task_specific_params
class GoNearPointOnObject(TypedDict):
    target_obj_in_3d: Dict[str, float]
    possible_points_on_target_in_first_frame: List[Tuple[float, float]]
    object_type: str
    object_id: str
    which_camera: Literal["nav", "manip"]


@register_task_specific_params
class PickupPointOnObject(TypedDict):
    target_obj_in_3d: Dict[str, float]
    possible_points_on_target_in_first_frame: List[Tuple[float, float]]
    object_type: str
    object_id: str
    which_camera: Literal["nav", "manip"]


@register_task_specific_params
class ConditionalNav(TypedDict):
    condition: bool
    condition_synset: str
    condition_object_ids: List[str]
    room_type: Optional[str]
    actual_condition_type: str
    actual_condition_state: bool

    positive_nav_type: str  # room, object
    positive_dest_synset: str
    positive_dest_ids: List[str]
    broad_positive_dest_ids: List[str]

    negative_nav_type: str  # room, object
    negative_dest_synset: str
    negative_dest_ids: List[str]
    broad_negative_dest_ids: List[str]


@register_task_specific_params
class ExploreObjects(TypedDict):
    pass


@register_task_specific_params
class LeaveAndReturn(TypedDict):
    min_distance_to_travel: float


@register_task_specific_params
class SimpleExploreHouse(ExploreHouse):
    num_rooms_in_house: int


def get_task_relevant_synsets(task_spec: Dict[str, Any]) -> List[str]:
    """Given an input task spec, returns a list of all synsets relevant to that task's success."""
    synsets = set()
    for k, v in task_spec.items():
        if "synset" in k:
            if k.endswith("synset_to_object_ids"):
                assert isinstance(v, Dict)
                synsets.update(v.keys())
            elif k in ["synsets", "reference_synsets"]:
                assert isinstance(v, Sequence)
                synsets.update(v)
                positive_dest_synset: str
            elif k in [
                "dest_receptacle_synset",
                "dest_synset",
                "condition_synset",
                "positive_dest_synset",
                "negative_dest_synset",
            ]:
                assert isinstance(v, str)
                synsets.add(v)
            else:
                raise NotImplementedError
    return list(synsets)


def get_train_tail_categories(n=4):
    global _TRAIN_TAIL_CATEGORIES

    if _TRAIN_TAIL_CATEGORIES is None:
        synset_counts = Counter(
            ann["synset"] for ann in _OBJAVERSE_ANNOTATIONS.values() if ann.get("split") == "train"
        )
        result = [synset for synset, count in synset_counts.items() if count < n]
        _TRAIN_TAIL_CATEGORIES = set(result + list(get_manual_skip_synsets()))

    return _TRAIN_TAIL_CATEGORIES


def get_manual_skip_synsets():
    openvocab_objnav_filter_list = [
        "fire_extinguisher.n.01",
        "map.n.01",
        "sponge.n.01",
        "shell.n.10",
        "remote_control.n.01",
        "tube.n.01",
        "bottlecap.n.01",
        "badge.n.01",
        "picture.n.05",
        "lotion.n.01",
        "disk_drive.n.01",
        "globe.n.03",
        "card.n.02",
        "clock.n.01",
        "collectible.n.01",
        "poster.n.01",
        "soap.n.01",
        "computer_game.n.01",
        "fossil.n.02",
        "router.n.02",
        "frame.n.11",
        "photograph.n.01",
        "marble.n.01",
        "cupboard.n.01",
        "disk.n.02",
    ]
    fetchtype_filter_list = [
        "automaton.n.02",
        "badge.n.01",
        "card.n.02",
        "checkerboard.n.01",
        "cigarette.n.01",
        "collectible.n.01",
        "computer_game.n.01",
        "disk.n.02",
        "disk_drive.n.01",
        "display.n.03",
        "faucet.n.01",
        "fictional_character.n.01",
        "figurine.n.01",
        "fork.n.01",
        "fossil.n.02",
        "gem.n.02",
        "grenade.n.01",
        "hand_blower.n.01",
        "jewelry.n.01",
        "keyboard.n.01",
        "lid.n.02",
        "light.n.02",
        "lotion.n.01",
        "magazine.n.01",
        "map.n.01",
        "marble.n.01",
        "mouse.n.04",
        "mousepad.n.01",
        "organism.n.01",
        "paddle.n.04",
        "paper_towel.n.01",
        "photograph.n.01",
        "picture.n.05",
        "pipe.n.01",
        "pipe.n.03",
        "poster.n.01",
        "receipt.n.02",
        "remote_control.n.01",
        "ring.n.02",
        "rock.n.01",
        "router.n.02",
        "sculpture.n.01",
        "shell.n.10",
        "tool.n.01",
        "trading_card.n.01",
        "tube.n.01",
        "videotape.n.01",
        "wristband.n.01",
    ]
    return set(openvocab_objnav_filter_list + fetchtype_filter_list)
