import sys


def get_mixture_by_name(name):
    return getattr(sys.modules[__name__], name, [name])


FIRST_SMALL_BATCH = [
    "ConditionalNav",
    "ExploreHouse",
    "Fetch2RoomAffordance",
    "Fetch2RoomType",
    "ObjectNavAffordance",
    "ObjectNavLocalRef",
    "ObjectNavMulti",
    "ObjectNavRelAttribute",
    "ObjectNavRoom",
    "ObjectNavType",
    "PickupAffordance",
    "PickupType",
    "RoomNav",
]

OBJECT_NAV_TYPE_DIFFICULTIES = [
    # "EasyObjectNavType",
    "ObjectNavType",
    "HardObjectNavType",
]

ONLY_OBJECTNAV_TYPE = ["ObjectNavType"]

ONLY_FETCH_TYPE = ["FetchType"]

EXPLORE_NAV_3TASK_MIX = ["ObjectNavType", "SimpleExploreHouse", "ExploreObjectNavType"]

FPO_v0_1 = ["FetchType", "PickupType", "ObjectNavType"]

EXPLORE_NAV_MIX = ["object_nav_v0.6", "explore_house_v0.8"]

TEST_ON_CPU = ["object_nav_v0.10", "pickup_v0.9"]

FETCHING_v0_1 = ["Fetch2SurfaceType", "FetchType"]

FPO_v0_2 = [
    "FetchType",
    "PickupType",
    "ObjectNavType",
    "Fetch2SurfaceType",
    "FetchType",
    "Fetch2RoomType",
]
SMALL_MULTITASK = [
    "RoomNav",
    "ObjectNavType",
    "ObjectNavRoom",
    "PickupType",
    "FetchType",
    "Fetch2RoomType",
    "ExploreHouse",
    "SimpleExploreHouse",
]

SMALL_MULTITASK_PRETRAIN = [
    "RoomNav",
    "ObjectNavType",
    "ObjectNavRoom",
    "PickupType",
    "FetchType",
    "Fetch2RoomType",
]

SMALL_SEVEN_MULTITASK = [
    "ConditionalNav",
    "RoomNav",
    "ObjectNavType",
    "ObjectNavRoom",
    "PickupType",
    "ExploreHouse",
    "SimpleExploreHouse",
]


SMALL_MULTITASK_2 = [
    "ObjectNavType",
    "ObjectNavRoom",
    "RoomNav",
    "PickupType",
    "ConditionalNav",
    "ExploreHouse",
    "SimpleExploreHouse",
]

ALL_TASKS = [
    "FetchType",
    "ObjectNavType",
    "PickupType",
    "ConditionalNav",
    "ExploreHouse",
    "ExploreObjects",
    "RoomNav",
    "ObjectNavRoom",
    "ObjectNavAffordance",
    "ObjectNavLocalRef",
    "ObjectNavOpenVocab",
    "ObjectNavRelAttribute",
    "PickupAffordance",
    "PickupOpenVocab",
    "SimpleExploreHouse",
    "ObjectNavRoomMulti",
    "ObjectNavMofN",
    "ObjectNavMulti",
    "FetchAffordance",
    "FetchLocalRef",
    "FetchObjRoom",
    "FetchOpenVocab",
    "FetchRelAttribute",
    "Fetch2RoomType",
    "Fetch2RefAffordance",
    "Fetch2RefObjRoom",
    "Fetch2RefOpenVocab",
    "Fetch2RefRelAttribute",
    "Fetch2RefType",
    "Fetch2RoomAffordance",
    "Fetch2RoomLocalRef",
    "Fetch2RoomObjRoom",
    "Fetch2RoomOpenVocab",
    "Fetch2RoomRelAttribute",
    "Fetch2SurfaceAffordance",
    "Fetch2SurfaceLocalRef",
    "Fetch2SurfaceMultiObject",
    "Fetch2SurfaceObjRoom",
    "Fetch2SurfaceOpenVocab",
    "Fetch2SurfaceRelAttribute",
    "Fetch2SurfaceType",
]

EASY_STUFF = [
    "EasyObjectNavType",
    "EasyFetchType",
]

OBJECT_NAV_TYPE_DIFFICULTIES = [
    "ObjectNavType",
    # "EasyObjectNavType",
    "HardObjectNavType",
]

ABLATION_TRAIN_TASKS = [
    "ObjectNavType",
    "ObjectNavRoom",
    "ObjectNavRelAttribute",
    "ObjectNavOpenVocab",
    "ObjectNavAffordance",
    "PickupType",
    "FetchType",
    "Fetch2RoomType",
    "RoomNav",
    "SimpleExploreHouse",
    "EasyFetchType",
    "EasyObjectNavType",
]

ABLATION_EVAL_TASKS = [
    "ObjectNavType",
    "ObjectNavRoom",
    "RoomNav",
    "PickupType",
    "FetchType",
    "Fetch2RoomType",
    "SimpleExploreHouse",
    "ObjectNavRelAttribute",
    "ObjectNavOpenVocab",
    "ObjectNavAffordance",
]

ABLATION_PLUS_EASY_EVAL_TASKS = ABLATION_EVAL_TASKS + ["EasyFetchType", "EasyObjectNavType"]

DifferentLevelObjectNav = [
    "ObjectNavType",
    "EasyObjectNavType",
    "HardObjectNavType",
    "HardExploreObjectNavType",
    "OldExploreObjectNavType300k",
]


SMALL_BENCH_TRAIN_TASKS = [
    "ObjectNavType",
    "ObjectNavRoom",
    "ObjectNavRelAttribute",
    "ObjectNavAffordance",
    "PickupType",
    "FetchType",
    "Fetch2RoomType",
    "RoomNav",
    "SimpleExploreHouse",
    "Fetch2SurfaceType",
    "EasyFetchType",
    "EasyObjectNavType",
]

SMALL_BENCH_EVAL_TASKS = [
    "ObjectNavType",
    "ObjectNavRoom",
    "ObjectNavRelAttribute",
    "ObjectNavAffordance",
    "PickupType",
    "FetchType",
    "Fetch2RoomType",
    "RoomNav",
    "SimpleExploreHouse",
    "Fetch2SurfaceType",
]

ABLATION_EVAL_TEMP_TASKS = [
    "Fetch2RoomType",
    "SimpleExploreHouse",
    "EasyFetchType",
    "EasyObjectNavType",
]

ABLATION_FOUR_TASKS = [
    "ObjectNavType",
    "PickupType",
    "FetchType",
    "SimpleExploreHouse",
]

ABLATION_TEN_TASKS = [
    "ObjectNavType",
    "ObjectNavRoom",
    "ObjectNavRelAttribute",
    "ObjectNavAffordance",
    "RoomNav",
    "PickupType",
    "FetchType",
    "Fetch2RoomType",
    "Fetch2SurfaceType",
    "SimpleExploreHouse",
]

ON_VARIATIONS = [
    "ObjectNavType",
    "ObjectNavRoom",
    "ObjectNavRelAttribute",
    "ObjectNavAffordance",
    "ObjectNavLocalRef",
    "ObjectNavOpenVocab",
    "RoomNav",
]

QUICK_TWO_TASKS = ["FetchType", "ObjectNavType"]

QUICK_TWO_TASKS_EVAL = ["FetchType", "ObjectNavType", "EasyFetchType", "EasyObjectNavType"]

QUICK_FOUR_TASKS = ["FetchType", "ObjectNavType", "EasyFetchType", "EasyObjectNavType"]

PURE_OBJECTNAVTYPE = ["ObjectNavType", "EasyObjectNavType"]

QUICK_TWO_TASKS_EVAL = ["FetchType", "ObjectNavType", "EasyFetchType"]

FetchAndObjectNav = ["FetchType", "ObjectNavType", "EasyFetchType", "EasyObjectNavType"]

QUICK_FOUR_TASKS_AND_SIMPLE_EXPLORE = QUICK_FOUR_TASKS + ["SimpleExploreHouse"]

LEFTOUT_ON_VAR = ["ObjectNavOpenVocab", "ObjectNavLocalRef"]

ON_VARIATIONS = [
    "ObjectNavType",
    "ObjectNavRoom",
    "ObjectNavRelAttribute",
    "ObjectNavAffordance",
    "ObjectNavLocalRef",
    "ObjectNavOpenVocab",
    "RoomNav",
]


ALL_POINT_TASKS = [
    "GoToPointOnFloor",
    "GoNearPointOnObject",
    "PickupPointOnObject",
    "DropOnPoint",
]

NAV_POINT_TASKS = [
    "GoNearPointOnObject",
    "GoToPointOnFloor",
]
