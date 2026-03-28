import copy
import random
from collections import Counter, defaultdict
from typing import (
    List,
    Literal,
    Optional,
    Dict,
    Any,
    Union,
    Sequence,
    Callable,
    Tuple,
    Set,
    TYPE_CHECKING,
)

import numpy as np
from utils.type_utils import get_train_tail_categories
from utils.data_generation_utils.object_filter_utils import (
    get_filters,
    apply_filters_on_potential_targets,
)
from utils.synsets.hypernyms import (
    generate_all_hypernyms_with_exclusions,
    filter_synsets_to_remove_hyponyms,
    get_hyponyms_of_synsets,
)
from environment.vida_objects import VIDAObject
from utils.data_generation_utils.exception_utils import (
    TaskSamplerException,
    HouseInvalidForTaskException,
)
from utils.data_generation_utils.navigation_utils import (
    get_room_id_from_location,
    object_ids_in_closed_receptacles,
)
from utils.data_generation_utils.randomization_utils import weighted_random_permutation_from_counts


if TYPE_CHECKING:
    from environment.stretch_controller import StretchController
    from utils.data_generation_utils.object_filter_utils import ObjectFilter

DEFAULT_PROB_RANDOMIZE_MATERIALS = 0.8


def get_objects_of_valid_synsets(
    controller: "StretchController",
    object_synsets_subset: Optional[List[str]] = None,
):
    # List object ids that cannot be seen:
    hidden_oids = object_ids_in_closed_receptacles(controller)

    all_objs = controller.get_objects()
    all_synsets_in_scene = set(o["synset"] for o in all_objs)
    valid_object_synsets = filter_synsets_to_remove_hyponyms(list(all_synsets_in_scene))
    if object_synsets_subset is not None:
        valid_object_synsets = set(valid_object_synsets) & set(object_synsets_subset)

    all_valid_hyponyms = get_hyponyms_of_synsets(valid_object_synsets, return_strings=True)

    valid_objects = [obj for obj in all_objs if obj["synset"] in all_valid_hyponyms]

    return {
        "all_objs": all_objs,
        "all_synsets_in_scene": all_synsets_in_scene,
        "valid_objs": valid_objects,
        "valid_object_synsets": set(valid_object_synsets),
        "hidden_oids": hidden_oids,
    }


def get_hypernym_to_objects(
    all_objs: Sequence[VIDAObject], valid_object_synsets: Set[str], excluded_hypernyms: Set[str]
):
    hypernym_to_objects = defaultdict(list)

    synset_to_hypernyms_cache = {}
    for obj in all_objs:
        if obj["synset"] not in synset_to_hypernyms_cache:
            hypernyms = generate_all_hypernyms_with_exclusions(
                obj["synset"], excluded=excluded_hypernyms
            )
            if obj["synset"] not in valid_object_synsets:
                base_hypernyms = valid_object_synsets & {h.name() for h in hypernyms}
                hypernyms = set()
                for bh in base_hypernyms:
                    if bh not in synset_to_hypernyms_cache:
                        synset_to_hypernyms_cache[bh] = generate_all_hypernyms_with_exclusions(
                            bh, excluded=excluded_hypernyms
                        )
                    hypernyms.update(synset_to_hypernyms_cache[bh])
            synset_to_hypernyms_cache[obj["synset"]] = list(hypernyms)

        for hypernym in synset_to_hypernyms_cache[obj["synset"]]:
            hypernym_to_objects[hypernym.name()].append(copy.deepcopy(obj))

    return hypernym_to_objects


def get_hypernym_instances_with_multiple(
    all_objs: Sequence[VIDAObject],
    valid_object_synsets: Set[str],
    excluded_hypernyms: Set[str],
    min_len: int = 2,
):
    return {
        s: l
        for s, l in get_hypernym_to_objects(
            all_objs=all_objs,
            valid_object_synsets=valid_object_synsets,
            excluded_hypernyms=excluded_hypernyms,
        ).items()
        if len(l) >= min_len
    }


def get_reachable_positions_3d_vectorized(controller: "StretchController") -> np.ndarray:
    # Convert all reachable positions to pixel space
    valid_locations_on_floor = controller.get_reachable_positions()
    valid_locations = np.array([[x["x"], x["y"], x["z"]] for x in valid_locations_on_floor])
    return valid_locations






def sample_object_synset_and_increment_counter(
    controller: "StretchController",
    object_synsets: List[str],
    object_synset_counter: Optional[Counter],
    filter_pickupable: bool,
    filter_by_height: bool,
    filter_navigable: bool,
    dest_callback: Optional[Callable[[str, Sequence[Dict[str, Any]]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    d = get_objects_of_valid_synsets(
        controller=controller,
        object_synsets_subset=object_synsets,
    )
    valid_object_synsets = d["valid_object_synsets"]

    if len(valid_object_synsets) == 0:
        raise TaskSamplerException(
            f"Could not find any objects of type {object_synsets} in the current scene."
        )

    filters = get_filters(
        filter_pickupable=filter_pickupable,
        filter_by_height=filter_by_height,
        filter_navigable=filter_navigable,
    )

    synset, oids, objs, broad_oids, dest_callback_res = pick_synset_and_increment_counter(
        controller=controller,
        object_synsets=valid_object_synsets,
        counter=object_synset_counter if object_synset_counter is not None else Counter(),
        filters=filters,
        dest_callback=dest_callback,
        hidden_oids=d["hidden_oids"],
    )

    if len(oids) == 0:
        raise TaskSamplerException(
            f"While some objects of types {valid_object_synsets} exist in the scene,"
            f" none satisfied given filters"
            f" (filter_pickupable={filter_pickupable},"
            f" filter_height={filter_by_height},"
            f" filter_reachable={filter_navigable})."
        )

    res = {"object_type": synset, "object_ids": oids, "objects": objs, "broad_ids": broad_oids}

    if dest_callback_res is not None:
        res["__dest_callback_result__"] = dest_callback_res

    return res





def sample_room_type_not_containing_target_objects(
    current_room_polymap_and_type,
    current_house_index: int,
    target_objects: Sequence[Dict[str, Any]],
):
    # find a room type that exists, make sure it is not similar to the current room type and
    # add all possible rooms to the task info,
    room_polymap, room_types = current_room_polymap_and_type
    all_source_room_types = [
        room_types[get_room_id_from_location(room_polymap, obj["position"])]
        for obj in target_objects
    ]

    other_room_types = {
        room_type
        for room_id, room_type in room_types.items()
        if room_type not in all_source_room_types
    }
    if len(other_room_types) == 0:
        raise TaskSamplerException(
            f"No room types were found not containing {target_objects} in house {current_house_index}"
            f" (all_source_room_types={all_source_room_types})"
        )
    dest_room_type = random.choice(list(other_room_types))
    dest_room_ids = [room_id for room_id in room_types if room_types[room_id] == dest_room_type]
    return dict(
        room_types=list(set(all_source_room_types)),
        dest_room_type=dest_room_type,
        dest_room_ids=dest_room_ids,
    )




def sample_open_vocab_object_and_increment_counter(
    controller: "StretchController",
    object_synsets: List[str],
    object_synset_counter: Optional[Counter],
    filter_pickupable: bool,
    filter_by_height: bool,
    filter_navigable: bool,
    dest_callback: Callable,
):
    if object_synset_counter is None:
        object_synset_counter = Counter()

    # Don't take into account objects that cannot be seen or are not pickupable:
    hidden_oids = object_ids_in_closed_receptacles(controller)
    all_objs = controller.get_objects()

    filters = get_filters(
        filter_pickupable=filter_pickupable,
        filter_by_height=filter_by_height,
        filter_navigable=filter_navigable,
    )
    # all_objs = [obj for obj in all_objs if obj["objectId"] not in hidden_oids]
    all_target_objs = [
        obj for obj in all_objs if obj["synset"] in object_synsets and obj["isObjaverse"]
    ]

    synset_to_objects = defaultdict(list)
    for obj in all_target_objs:
        synset_to_objects[obj["synset"]].append(obj)

    asset_id_to_targets = defaultdict(list)
    for obj in all_target_objs:
        asset_id_to_targets[obj["assetId"]].append(obj)

    # TODO we might also want to constrain to object types for which more than one instance of
    #  the type is present in the scene - not sure it's really required, since open vocabulary
    #  already adds complexity to the task.

    # TODO we should ensure the description doesn't closely resemble any other in house

    # use the counter to balance it out
    for synset in weighted_random_permutation_from_counts(
        l=list(synset_to_objects.keys()), counts=object_synset_counter
    ):
        if synset in get_train_tail_categories():
            continue

        potential_objects = synset_to_objects[synset]

        for potential_object in potential_objects:
            true_object_synset = potential_object["synset"]
            object_asset_id = potential_object["assetId"]

            if len(asset_id_to_targets[object_asset_id]) != 1:
                print(f"Skipping multi-instance asset {object_asset_id} ({true_object_synset})")
                continue

            # Make sure all filters are passed for chosen instance
            passing_objs = apply_filters_on_potential_targets(
                [potential_object], controller, filters, hidden_oids
            )
            if len(passing_objs) != 1:
                continue

            try:
                dest_dict = dest_callback(
                    all_objs=all_objs,
                    potential_objects=[potential_object],
                    potential_ids={potential_object["objectId"]},
                    hidden_oids=hidden_oids,
                )
            except:
                continue

            if dest_dict is None:
                continue

            object_synset_counter[synset] += 1

            return dict(
                synsets=[true_object_synset],
                synset_to_object_ids={true_object_synset: [potential_object["objectId"]]},
                broad_synset_to_object_ids={true_object_synset: [potential_object["objectId"]]},
                uid=object_asset_id,
                **dest_dict,
            )

    raise HouseInvalidForTaskException("No valid params found")


def pick_synset_and_increment_counter(
    controller: "StretchController",
    object_synsets: Union[Set[str], Sequence[str]],
    counter: Dict[str, int],
    filters: Optional[List["ObjectFilter"]],
    dest_callback: Optional[Callable[[str, Sequence[Dict[str, Any]]], Dict[str, Any]]],
    hidden_oids: Sequence[str],
    min_objects: int = 1,
) -> Tuple[
    Optional[str], List[str], List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]
]:
    if len(object_synsets) == 0:
        return None, [], [], [], None

    if filters is None:
        filters = []

    # use the counter to balance it out
    for synset in weighted_random_permutation_from_counts(l=object_synsets, counts=counter):
        if synset in get_train_tail_categories():
            continue

        all_obj_of_synset = controller.get_all_objects_of_synset(synset, include_hyponyms=True)

        filtered_obj_of_synset = apply_filters_on_potential_targets(
            all_obj_of_synset, controller, filters, hidden_oids
        )

        if len(filtered_obj_of_synset) >= min_objects:
            callback_res = None
            if dest_callback is not None:
                # Note: callback might raise if no valid configuration is possible
                try:
                    callback_res = dest_callback(synset, all_obj_of_synset)
                except TaskSamplerException:
                    continue

            counter[synset] += 1
            return (
                synset,
                [o["objectId"] for o in filtered_obj_of_synset],
                filtered_obj_of_synset,
                [o["objectId"] for o in all_obj_of_synset],
                callback_res,
            )

    return None, [], [], [], None
