from collections import Counter
import random
from typing import Dict, Any, Sequence, Set

import numpy as np

from tasks.base_task_sampler import BaseTaskSampler
from utils.data_generation_utils.exception_utils import (
    TaskSamplerException,
    HouseInvalidForTaskException,
)
from utils.data_generation_utils.navigation_utils import filtered_starting_positions
from utils.data_generation_utils.task_spec_sampling_utils import (
    sample_object_synset_and_increment_counter,
    sample_open_vocab_object_and_increment_counter,
)

from utils.constants.stretch_initialization_utils import HORIZON
from utils.type_utils import Vector3, AgentPose
from utils.constants.objaverse_data_dirs import OBJAVERSE_ASSETS_VERSION


class ObjectNavTaskSampler(BaseTaskSampler):
    task_type_str = "ObjectNavType"
    TELEPORT_THEN_SAMPLE = OBJAVERSE_ASSETS_VERSION not in ["2025_06_10"]

    def select_and_teleport_to_start(self, room_id_to_fsps, max_tries=10, params=None):
        if self.TELEPORT_THEN_SAMPLE:
            return super().select_and_teleport_to_start(room_id_to_fsps, params=params)

        object_type = params["synsets"][0]
        flat_id_list = [*params["synset_to_object_ids"][object_type]]

        room_order = list(room_id_to_fsps.keys())
        random.shuffle(room_order)

        for room_id in room_order:
            controller_reachable_positions = room_id_to_fsps[room_id]
            dists = [np.inf] * len(controller_reachable_positions)

            for target_object in flat_id_list:
                chosen_pose = self.controller.get_object_position(target_object)

                def position_dist_from_dict(dict1, dict2):
                    return abs(dict1["x"] - dict2["x"]) + abs(dict1["z"] - dict2["z"])

                dists = [
                    min(d, position_dist_from_dict(x, chosen_pose))
                    for x, d in zip(controller_reachable_positions, dists)
                ]

                # Actual threshold is 3 for planned path, but we can also be far
                # from the target to succeed, so we might prefer to use a tighter threshold here.
                # We can also need a Manhattan distance to the goal assuming the presence of many
                # obstacles in the room, so what shall we do? The optimistic one, aka Manhattan, and
                # let the planner decide
                controller_reachable_positions = [
                    x for x, d in zip(controller_reachable_positions, dists) if d >= 3.0
                ]
                dists = [d for d in dists if d >= 3.0]

                if not controller_reachable_positions:
                    break

            if not controller_reachable_positions:
                break

            to_sort = [(d, x) for x, d in zip(controller_reachable_positions, dists)]
            all_poses = [
                x[1]
                for x in sorted(to_sort, key=lambda x: x[0], reverse=True)[
                    : max(len(to_sort) // 2, 1)
                ]
            ]
            random.shuffle(all_poses)

            # random.shuffle(controller_reachable_positions)
            for pose in all_poses[:max_tries]:
                starting_pose = AgentPose(
                    position=Vector3(x=pose["x"], y=pose["y"], z=pose["z"]),
                    rotation=Vector3(x=0, y=random.choice(list(range(0, 360, 30))), z=0),
                )
                event = self.controller.teleport_agent(**starting_pose)

                if not event:
                    continue

                return starting_pose, False

        return None, True  # True to resample task params and try different objects

    def sample_task_parameters(self) -> Dict[str, Any]:
        sampled_source_dict = sample_object_synset_and_increment_counter(
            controller=self.controller,
            object_synsets=self.target_object_types,
            object_synset_counter=self.object_synset_counter,
            filter_pickupable=False,
            filter_by_height=True,
            filter_navigable=True,
        )

        return dict(
            synsets=[sampled_source_dict["object_type"]],
            synset_to_object_ids={
                sampled_source_dict["object_type"]: sampled_source_dict["object_ids"]
            },
            broad_synset_to_object_ids={
                sampled_source_dict["object_type"]: sampled_source_dict["broad_ids"]
            },
        )


class EasyObjectNavTaskSampler(ObjectNavTaskSampler):
    task_type_str = "EasyObjectNavType"
    TELEPORT_THEN_SAMPLE = False

    def select_and_teleport_to_start(self, room_id_to_fsps, max_tries=2, params=None):
        object_type = params["synsets"][0]
        flat_id_list = [*params["synset_to_object_ids"][object_type]]

        random.shuffle(flat_id_list)
        controller_reachable_positions = self.controller.get_reachable_positions()
        for target_object in flat_id_list[:max_tries]:

            def position_dist_from_dict(dict1, dict2):
                return ((dict1["x"] - dict2["x"]) ** 2 + (dict1["z"] - dict2["z"]) ** 2) ** 0.5

            chosen_pose = self.controller.get_object_position(target_object)

            # Get closest points
            sorted_reachable_positions = sorted(
                controller_reachable_positions,
                key=lambda x: position_dist_from_dict(x, chosen_pose),
            )

            begin_index = 100
            end_index = begin_index + 60  # TODO KE: IS THIS GOOD?

            begin_index = max(0, begin_index)
            end_index = min(len(sorted_reachable_positions), end_index)

            poses = sorted_reachable_positions[begin_index:end_index]
            random.shuffle(poses)

            for pose in poses:
                starting_pose = AgentPose(
                    position=Vector3(x=pose["x"], y=pose["y"], z=pose["z"]),
                    rotation=Vector3(x=0, y=random.choice(list(range(0, 360, 30))), z=0),
                )
                event = self.controller.teleport_agent(**starting_pose)

                if not event:
                    continue

                params["synset_to_object_ids"] = {object_type: [target_object]}
                return starting_pose, False

        return None, True  # True to resample task params and try different objects


class ObjectNavOpenVocabTaskSampler(BaseTaskSampler):
    task_type_str = "ObjectNavOpenVocab"

    def sample_task_parameters(self) -> Dict[str, Any]:
        def dest_callback(
            all_objs: Sequence[Dict[str, Any]],
            potential_objects: Sequence[Dict[str, Any]],
            potential_ids: Set[str],
            hidden_oids: Sequence[str],
        ):
            return {}

        return sample_open_vocab_object_and_increment_counter(
            controller=self.controller,
            object_synsets=self.target_object_types,
            object_synset_counter=self.object_synset_counter,
            filter_pickupable=False,
            filter_by_height=True,
            filter_navigable=False,  # TODO: Currently would be too slow not to do this
            dest_callback=dest_callback,
        )


class HardObjectNavTaskSampler(ObjectNavTaskSampler):
    task_type_str = "HardObjectNavType"
    TELEPORT_THEN_SAMPLE = False

    def select_and_teleport_to_start(self, room_id_to_fsps, max_tries=10, params=None):
        if len(room_id_to_fsps) < 4:
            raise HouseInvalidForTaskException(
                "Difficult tasks must be in houses with at least 4 rooms"
            )
        object_type = params["synsets"][0]
        flat_id_list = [*params["broad_synset_to_object_ids"][object_type]]
        controller_reachable_positions = self.reachable_positions
        filtered_reachable_positions = filtered_starting_positions(controller_reachable_positions)

        # Get positions of all target objects
        object_positions = [
            self.controller.get_object_position(target_object) for target_object in flat_id_list
        ]

        def position_dist_from_dict(dict1, dict2):
            return ((dict1["x"] - dict2["x"]) ** 2 + (dict1["z"] - dict2["z"]) ** 2) ** 0.5

        # For each reachable position, find the minimum distance to any target object
        min_dists = []
        for pos in filtered_reachable_positions:
            min_dist = min(position_dist_from_dict(pos, obj_pos) for obj_pos in object_positions)
            min_dists.append(min_dist)

        # Sort reachable positions by minimum distance in descending order
        sorted_reachable_positions = [
            pos
            for _, pos in sorted(
                zip(min_dists, filtered_reachable_positions), key=lambda pair: pair[0], reverse=True
            )
        ]

        begin_index = 0
        end_index = begin_index + 100

        begin_index = max(0, begin_index)
        end_index = min(len(sorted_reachable_positions), end_index)

        poses = sorted_reachable_positions[begin_index:end_index]
        random.shuffle(poses)

        for _ in range(max_tries):
            pose = poses.pop(0)
            starting_pose = AgentPose(
                position=Vector3(x=pose["x"], y=pose["y"], z=pose["z"]),
                rotation=Vector3(x=0, y=random.choice(list(range(0, 360, 30))), z=0),
            )
            event = self.controller.teleport_agent(**starting_pose)

            if not event:
                continue

            return starting_pose, False

        return None, True  # True to resample task params and try different objects
