from typing import Protocol, Dict, Any, List, Sequence, Optional

import numpy as np

from environment.stretch_controller import StretchController
from utils.data_generation_utils.bbox_utils import get_basis_for_3d_box, get_min_object_height
from utils.type_utils import get_train_tail_categories
from vida_constants import (
    GO_NEAR_POINT_SAMPLING_MIN_PIXEL_VISIBLE_THRESHOLD,
    GO_NEAR_POINT_SAMPLING_MIN_WH_THRESHOLD,
    GO_NEAR_POINT_SUCCESS_DIST_THRESHOLD,
    MIN_HEIGHT_OF_OBJECT_TO_SAMPLE_FOR_PICKUP,
    PICKUP_POINT_SAMPLING_DIST_THRESHOLD,
)

MAX_HEIGHT_OF_NAV_OBJECT_TO_SAMPLE = 1.1
MAX_BBOX_VOLUME_FOR_PICKUP = 0.009  # 30cm cube. Not currently used
MAX_MAX_DIMENSION_FOR_PICKUP = 0.5  # 50 cm in max dimension
MAX_MIN_DIMENSION_FOR_PICKUP = 0.2  # 20 cm (approx width of gripper)
MIN_LARGEST_FACE_DIAGONAL = 0.1  # min diagonal of an object's bbox to be considered a demo target
MIN_MIDDLE_DIMENSION = 0.04  # 4 cm in the middle dimension of the bbox


class ObjectFilter(Protocol):
    def __call__(self, controller: "StretchController", o: Dict[str, Any]) -> bool: ...


def is_pickupable(controller: "StretchController", o: Dict[str, Any]):
    if not o["pickupable"]:
        return False  # can't compute the bounding box for a floor, go team
    obj_bbox = get_basis_for_3d_box(o)
    return max(obj_bbox[1]) < MAX_MAX_DIMENSION_FOR_PICKUP


def is_below_height(controller: "StretchController", o: Dict[str, Any]):
    return get_min_object_height(o) < MAX_HEIGHT_OF_NAV_OBJECT_TO_SAMPLE


def is_above_height(controller: "StretchController", o: Dict[str, Any]):
    return get_min_object_height(o) >= MIN_HEIGHT_OF_OBJECT_TO_SAMPLE_FOR_PICKUP


def if_within_distance(controller: "StretchController", o: Dict[str, Any]):
    current_agent_pose = controller.get_current_agent_position()
    object_position = o["position"]
    dist = (
        (current_agent_pose["x"] - object_position["x"]) ** 2
        + (current_agent_pose["z"] - object_position["z"]) ** 2
    ) ** 0.5
    return dist < PICKUP_POINT_SAMPLING_DIST_THRESHOLD


def if_further_than_distance(controller: "StretchController", o: Dict[str, Any]):
    current_agent_pose = controller.get_current_agent_position()
    object_position = o["position"]
    dist = (
        (current_agent_pose["x"] - object_position["x"]) ** 2
        + (current_agent_pose["z"] - object_position["z"]) ** 2
    ) ** 0.5
    return dist > (GO_NEAR_POINT_SUCCESS_DIST_THRESHOLD + 0.5)


def has_big_wide_mask(controller: "StretchController", o: Dict[str, Any], which_camera):
    mask = controller.get_segmentation_mask_of_object(
        object_id=o["objectId"], which_camera=which_camera
    )
    non_zeros = np.where(mask)
    if mask.sum() == 0:
        return False
    min_x, max_x = np.min(non_zeros[1]), np.max(non_zeros[1])
    min_y, max_y = np.min(non_zeros[0]), np.max(non_zeros[0])

    return (
        max_x - min_x >= GO_NEAR_POINT_SAMPLING_MIN_WH_THRESHOLD
        and max_y - min_y >= GO_NEAR_POINT_SAMPLING_MIN_WH_THRESHOLD
        and mask.sum() >= GO_NEAR_POINT_SAMPLING_MIN_PIXEL_VISIBLE_THRESHOLD
    )


def is_navigable(controller: "StretchController", o: Dict[str, Any]):
    return controller.does_some_shortest_path_to_object_exist(o["objectId"])


def is_diag_and_side_large(controller: "StretchController", o: Dict[str, Any]):
    # the goal is to avoid elongated objects like a stretched thread, where the largest and second-largest face
    # diagonals can be large, but two of the dimensions are almost negligible

    if o["objectType"] == "Floor":
        return False

    obj_basis, basis_mags = get_basis_for_3d_box(o)
    norms = sorted(basis_mags)
    largest_face_diag = np.sqrt(norms[1] ** 2 + norms[2] ** 2)
    middle_dimension = norms[1]

    return (
        largest_face_diag >= MIN_LARGEST_FACE_DIAGONAL and middle_dimension >= MIN_MIDDLE_DIMENSION
    )


def is_not_object_part(controller: "StretchController", o: Dict[str, Any]):
    return "___" not in o["objectId"]


def is_not_train_tail(controller: "StretchController", o: Dict[str, Any]):
    return o["synset"] not in get_train_tail_categories()


def get_filters(
    filter_pickupable: bool,
    filter_by_height: bool,
    filter_navigable: bool,
    filter_large_diag: bool = True,
    filter_object_parts: bool = True,
    filter_train_tail: bool = True,
    filter_by_distance: bool = False,
    filter_by_min_height: bool = False,
    filter_is_further_than_distance: bool = False,
    filter_has_big_wide_mask: bool = False,
) -> List[ObjectFilter]:
    filters = []
    if filter_pickupable:
        filters.append(is_pickupable)
    if filter_by_height:
        filters.append(is_below_height)
    if filter_by_min_height:
        filters.append(is_above_height)
    if filter_navigable:
        filters.append(is_navigable)
    if filter_large_diag:
        filters.append(is_diag_and_side_large)
    if filter_object_parts:
        filters.append(is_not_object_part)
    if filter_train_tail:
        filters.append(is_not_train_tail)
    if filter_by_distance:
        filters.append(if_within_distance)
    if filter_has_big_wide_mask:
        filters.append(
            lambda controller, o, which_camera: has_big_wide_mask(controller, o, which_camera)
        )
    if filter_is_further_than_distance:
        filters.append(if_further_than_distance)
    return filters


def apply_filters_on_potential_targets(
    potential_targets: Sequence[Dict[str, Any]],
    controller: Optional["StretchController"] = None,
    filters: Optional[Sequence[ObjectFilter]] = None,
    hidden_oids: Optional[Sequence[str]] = None,
    use_filters: bool = True,
    exclude_hidden_oids: bool = True,
    which_camera: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if filters is not None and use_filters:
        assert (
            controller is not None
        ), "Missing controller in call to `apply_filters_on_potential_targets`"
    return [
        o
        for o in potential_targets
        if (not exclude_hidden_oids or hidden_oids is None or o["objectId"] not in hidden_oids)
        and (
            not use_filters
            or filters is None
            or all(
                (
                    f(controller=controller, o=o)
                    if "which_camera" not in f.__code__.co_varnames
                    else f(controller=controller, o=o, which_camera=which_camera)
                )
                for f in filters
            )
        )
    ]
