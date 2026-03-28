import argparse
import configparser
import json
import math
import os
import platform
import sys
import time
import unittest
from typing import Literal, Dict, Optional, Union, Sequence

import canonicaljson
from environment.object_nav_sensors import (
    AgentParametersSensor,
    LastActionStrSensor, 
    LastActionSuccessSensor, 
    LastAgentLocationSensor, 
    TaskTemplatedTextSpecSensor, 
    TimeStepSensor, 
    TrajectorySensor
)
import prior
from allenact.base_abstractions.sensor import Sensor

from data_generation.sensors import (
    CameraFollowAgent,
    RawManipulationStretchDepthSensor,
    RawNavigationStretchDepthSensor,
    RawNavigationStretchRGBSensor,
    RawManipulationStretchRGBSensor,
)
from environment.action_spaces import (
    SPOCV1ActionSpace,
    AbstractActionSpace,
)

from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from utils.constants.object_constants import (
    OBJNAV_SYNSETS,
    AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET,
    TARGET_EXCLUDED_SYNSETS,
)
from utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_WIDTH,
    INTEL_CAMERA_HEIGHT,
    SAVE_DEPTH,
)
from utils.objaverse_utils import _OBJAVERSE_ANNOTATIONS
from utils.type_utils import AbstractTaskArgs
from vida_constants import ABS_PATH_OF_VIDA_DIR

# Get the BEAKER_ASSIGNED_MEMORY_BYTES environment variable
_BEAKER_MEMORY_BYTES = os.getenv("BEAKER_ASSIGNED_MEMORY_BYTES")
_MEM_PER_WORKER = 15 * 1024**3

NUM_WORKERS_ON_SINGLE_DEVICE = os.getenv("NUM_WORKERS_ON_SINGLE_DEVICE")

if NUM_WORKERS_ON_SINGLE_DEVICE is None:
    # Calculate the number of workers based on the memory amount
    if sys.platform == "linux":
        _BEAKER_MEMORY_BYTES = os.getenv("BEAKER_ASSIGNED_MEMORY_BYTES")
        if _BEAKER_MEMORY_BYTES is None:
            print(
                "Warning: BEAKER_ASSIGNED_MEMORY_BYTES environment variable not set on Linux. Defaulting to 3 workers."
            )
            NUM_WORKERS_ON_SINGLE_DEVICE = 3
        else:
            # Convert the memory amount to an integer
            _BEAKER_MEMORY_BYTES = int(_BEAKER_MEMORY_BYTES)
            print(f"Converted BEAKER_ASSIGNED_MEMORY_BYTES to integer: {_BEAKER_MEMORY_BYTES}")

            # Calculate the number of 15GiB increments in the memory amount
            NUM_WORKERS_ON_SINGLE_DEVICE = math.floor(_BEAKER_MEMORY_BYTES / _MEM_PER_WORKER)
            print(f"Calculated number of workers: {NUM_WORKERS_ON_SINGLE_DEVICE}")

    else:
        NUM_WORKERS_ON_SINGLE_DEVICE = 1
else:
    NUM_WORKERS_ON_SINGLE_DEVICE = int(NUM_WORKERS_ON_SINGLE_DEVICE)

print(f"Set NUM_WORKERS_ON_SINGLE_DEVICE to {NUM_WORKERS_ON_SINGLE_DEVICE}")

_DATASET_CACHE = {}


def get_house_dataset(
    house_dataset: Literal["objaverse", "procthor", "phone2proc"],
    debug=False,
    max_houses_per_split: Optional[Union[int, Dict[str, int]]] = None,
):
    key = (
        house_dataset,
        debug,
        canonicaljson.encode_canonical_json(max_houses_per_split).decode("utf-8"),
    )
    if key not in _DATASET_CACHE:
        if house_dataset == "objaverse":
            if max_houses_per_split is None and debug:
                max_houses_per_split = 100

            _DATASET_CACHE[key] = prior.load_dataset(
                "procthor-objaverse-internal",
                revision="local",
                # path_to_splits=OBJAVERSE_HOUSES_DIR,
                path_to_splits=None,
                split_to_path={
                    k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
                    for k in ["train", "val", "test"]
                },
                max_houses_per_split=max_houses_per_split,
            )
        elif house_dataset == "procthor":
            _DATASET_CACHE[key] = prior.load_dataset(
                dataset="procthor-100k",
                entity="roseh-ai2",
                revision="tiny" if debug else "100k-balanced-may03",
            )
        elif house_dataset == "phone2proc":
            _DATASET_CACHE[key] = prior.load_dataset(
                dataset="all-back-apartment-data",
            )
        else:
            raise NotImplementedError(f"Unknown dataset {house_dataset}")
    return _DATASET_CACHE[key]



def get_core_sensors(datagen=False):
    sensors = [
        RawNavigationStretchRGBSensor(
            uuid="raw_navigation_camera",
            width=INTEL_CAMERA_WIDTH,
            height=INTEL_CAMERA_HEIGHT,
        ),
        RawManipulationStretchRGBSensor(
            uuid="raw_manipulation_camera",
            width=INTEL_CAMERA_WIDTH,
            height=INTEL_CAMERA_HEIGHT,
        ),
        LastActionStrSensor(),
        LastActionSuccessSensor(),
        TimeStepSensor(uuid="time_ids", max_time_for_random_shift=0),
        TrajectorySensor(uuid="traj_index", max_idx=2048),
        TaskTemplatedTextSpecSensor(),
        LastAgentLocationSensor(),  # pyright: ignore[reportUndefinedVariable]
        AgentParametersSensor(datagen=datagen),

    ]
    if platform.system() == "Darwin":
        sensors.append(
            CameraFollowAgent(
                uuid="camera_follow_agent",
                width=INTEL_CAMERA_WIDTH,
                height=INTEL_CAMERA_HEIGHT,
            ),
        )
    return sensors



def get_core_task_args(max_steps: int, action_space: AbstractActionSpace, datagen: bool=False) -> AbstractTaskArgs: 
    return AbstractTaskArgs(
        sensors=get_core_sensors(datagen=datagen),
        action_space=action_space,
        max_steps=max_steps,
        reward_config=None,
    )


def add_extra_sensors_to_task_args(
    task_args: AbstractTaskArgs, extra_sensors: Optional[Sequence[Sensor]]
):
    if extra_sensors is None or len(extra_sensors) == 0:
        return

    core_sensor_dict = {x.uuid: x for x in task_args["sensors"]}

    for sensor in extra_sensors:
        if sensor.uuid in core_sensor_dict:
            print(f"WARNING: WE ARE REPLACING {core_sensor_dict[sensor.uuid]} WITH {sensor}")
            del core_sensor_dict[sensor.uuid]

    task_args["sensors"] = list(core_sensor_dict.values()) + list(extra_sensors)


def construct_action_space_from_args(action_space_type: str, action_space_subtype: str):
    config = configparser.ConfigParser()

    # get config name from args formatted to the config file name
    config_path = os.path.abspath(
        os.path.join(
            ABS_PATH_OF_VIDA_DIR,
            f"configs/dataset_generation/action_spaces/{action_space_type}.ini",
        )
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config.read(
        config_path
    )  # DOES NOT RAISE ERRORS IF config_path DOES NOT EXIST, HORRIBLE BEHAVIOR, BAD CONFIG

    action_space_subset = action_space_subtype

    if not config.has_section(action_space_subset):
        raise ValueError(
            f"Action space name '{action_space_subset}' does not exist in the config file."
        )

    action_space_settings = dict(config.items(action_space_subset))

    try:
        if action_space_settings["type"] == "spoc":
            action_space = SPOCV1ActionSpace()
        else:
            raise ValueError(f"Invalid action space type: {action_space_settings['type']}")
    except KeyError as e:
        raise ValueError(
            f"Missing key '{e.args[0]}' in the action space settings for '{action_space_subset}'"
        )

    return action_space


def log_task_arguments(task_arguments, json_file_path):
    if isinstance(task_arguments, list):
        task_arguments = " ".join(task_arguments)

    # Split on the first instance of '--'
    task_arguments = task_arguments.split("--", 1)

    # If there are two parts, the second part should be the arguments
    if len(task_arguments) == 2:
        task_arguments = "--" + task_arguments[1]

    task_arguments = task_arguments.split()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--sqs_queue_name", default=None)
    parser.add_argument("--metrics_json_dir", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--task_type", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--debug", default=None)
    parser.add_argument("--closed_type", default=None)
    parser.add_argument("--house_dataset", default=None)
    parser.add_argument("--action_space_type", default=None)
    parser.add_argument("--action_space_name", default=None)

    args = parser.parse_args(task_arguments)

    arguments_dict = vars(args)
    task_type = arguments_dict.get("task_type")

    # Prepare the data to be logged
    timestamp = time.time()

    # Check if the file exists
    if os.path.exists(json_file_path):
        # Load existing data
        with open(json_file_path, "r") as f:
            existing_data = json.load(f)

        # Check if the task type already exists in the data
        if task_type in existing_data:
            # Compare the current task's action_space_type and action_space_name with the ones in the existing data
            if existing_data[task_type]["arguments"].get("action_space_type") != arguments_dict.get(
                "action_space_type"
            ) or existing_data[task_type]["arguments"].get(
                "action_space_name"
            ) != arguments_dict.get(
                "action_space_name"
            ):
                raise Exception(
                    "The current task's action_space_type or action_space_name "
                    "are not the same as the ones in the existing data."
                )
            else:
                # Append the current timestamp to the list of timestamps
                existing_data[task_type]["timestamps"].append(timestamp)
        else:
            # If the task type does not exist in the data, add it
            existing_data[task_type] = {"timestamps": [timestamp], "arguments": arguments_dict}

        # Write back to the file
        with open(json_file_path, "w") as f:
            json.dump(existing_data, f)
    else:
        # Create a new JSON file with the data for the task type
        with open(json_file_path, "w") as f:
            json.dump({task_type: {"timestamps": [timestamp], "arguments": arguments_dict}}, f)


def get_task_args_and_object_synsets(
    task_type: str,
    max_steps: int,
    action_space: AbstractActionSpace,
    closed_type=False,
    extra_sensors: Sequence[Sensor] = tuple(),
):
    destination_object_types = []
    if closed_type:
        # note: existing filters will handle pickup vs nav objects
        # note also closing type will make OpenVocab a bit difficult to sample
        # because not as many appropriate candidate houses
        target_object_types = OBJNAV_SYNSETS
        if task_type.startswith("Fetch2Ref"):
            destination_object_types = target_object_types

    else:
        all_synsets = set(
            item.get("synset") for item in _OBJAVERSE_ANNOTATIONS.values() if "synset" in item
        ) | set(AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.values())

        target_object_types = [syn for syn in all_synsets if syn not in TARGET_EXCLUDED_SYNSETS]

        if task_type.startswith("Fetch2Ref"):
            destination_object_types = target_object_types

    task_args = get_core_task_args(max_steps=max_steps, action_space=action_space, datagen=True)

    if SAVE_DEPTH:
        extra_sensors = list(extra_sensors) + [
            RawNavigationStretchDepthSensor(
                uuid="raw_navigation_depth", width=INTEL_CAMERA_WIDTH, height=INTEL_CAMERA_HEIGHT
            ),
            RawManipulationStretchDepthSensor(
                uuid="raw_manipulation_depth", width=INTEL_CAMERA_WIDTH, height=INTEL_CAMERA_HEIGHT
            ),
        ]

    add_extra_sensors_to_task_args(task_args, extra_sensors)

    return task_args, target_object_types, destination_object_types


class TestLogTaskArguments(unittest.TestCase):
    def setUp(self):
        self.json_file_path = "test_log_task_arguments.json"
        self.task_arguments = [
            "--task_type",
            "test_task",
            "--action_space_type",
            "test_space",
            "--action_space_name",
            "test_name",
        ]

    def tearDown(self):
        if os.path.exists(self.json_file_path):
            os.remove(self.json_file_path)

    def test_log_task_arguments(self):
        # Call the function with the test arguments
        log_task_arguments(self.task_arguments, self.json_file_path)

        # Check if the file was created
        self.assertTrue(os.path.exists(self.json_file_path))

        # Load the data from the file
        with open(self.json_file_path, "r") as f:
            data = json.load(f)

        # Check if the data is correct
        self.assertIn("test_task", data)
        self.assertIn("timestamps", data["test_task"])
        self.assertIn("arguments", data["test_task"])
        self.assertEqual(data["test_task"]["arguments"]["task_type"], "test_task")
        self.assertEqual(data["test_task"]["arguments"]["action_space_type"], "test_space")
        self.assertEqual(data["test_task"]["arguments"]["action_space_name"], "test_name")

        different_task_arguments = [
            f"/opt/miniconda3/envs/vida2/bin/python not_a_real_script.py ",
            "--task_type",
            "test_task",
            "--action_space_type",
            "different_space",
            "--action_space_name",
            "different_name",
        ]
        with self.assertRaises(Exception):
            log_task_arguments(different_task_arguments, self.json_file_path)


if __name__ == "__main__":
    unittest.main()
