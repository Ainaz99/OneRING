import glob
import json
import math
import os
import pdb
import platform
import traceback
from typing import List, Sequence, Union
import cv2
from PIL import Image
import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from tqdm import tqdm
from copy import deepcopy

from utils.sensor_constant_utils import is_a_visual_sensor
from utils.string_utils import (
    convert_byte_to_string,
    json_templated_spec_to_dict,
    json_templated_to_NL_spec,
)


def video_quality_control(
    sample_id: str,
    sample_path: str,
    original_frames_len: int,
    original_actions_len: int,
    frames: torch.Tensor,
) -> bool:
    msg = "frames and actions do not match for sample id " + sample_id + sample_path
    try:
        assert original_frames_len == original_actions_len, print(
            msg,
            "original_frame",
            original_frames_len,
            "original actions",
            original_actions_len,
        )
    except Exception as e:
        print("In Lengths not equal")
        print(str(e))
        print(traceback.format_exc())
        print(msg)
        return False
    return True

class iLearnReader:
    def __init__(self, data_dir, subset, proc_idx=0, num_procs=1, max_samples=None, seed=123):
        self.data_dir = data_dir
        self.subset = subset
        self.proc_idx = proc_idx
        self.num_procs = num_procs
        self.max_samples = max_samples
        self.seed = seed
        self.house_id_to_sub_house_id_json = os.path.join(
            data_dir, f"house_id_to_sub_house_id_{self.subset}.json"
        )

        self.dataset_version = self.data_dir.split("/")[-1].split("_")[-1]

    def load_samples_for_proc_idx(self, proc_idx: int):
        # select house files based on the process id
        with open(self.house_id_to_sub_house_id_json, "r") as f:
            house_id_to_sub_house_id = json.load(f)
        # house_id_to_sub_house_id = json.load(open(self.house_id_to_sub_house_id_json, "r"))

        house_ids = sorted(list(house_id_to_sub_house_id.keys()))
        assert len(house_ids) > 0, f"{self.data_dir}/{self.subset} doesn't exist or has no houses"
        random.seed(self.seed)
        random.shuffle(house_ids)
        house_ids = [
            house_id for i, house_id in enumerate(house_ids) if i % self.num_procs == proc_idx
        ]
        # read samples from all the houses w/o reading in the sensors from hdf5
        print(
            f"Reading samples from {self.data_dir}/{self.subset} with process {proc_idx}/{self.num_procs}"
        )
        samples = []
        for house_id in tqdm(house_ids):
            house_dir = os.path.join(self.data_dir, self.subset, house_id)
            sensors_path = os.path.join(house_dir, "hdf5_sensors.hdf5")
            with h5py.File(sensors_path, 'r') as hdf:
                for sub_house_id in house_id_to_sub_house_id[house_id]:
                    sample_id = f"house={house_id},sub_house_id={sub_house_id}"
                    nav_cam_video = os.path.join(
                        house_dir, f"raw_navigation_camera__{sub_house_id}.mp4"
                    )
                    manip_cam_video = nav_cam_video.replace("navigation", "manipulation")
                
                    samples.append(
                            dict(
                                sample_id=sample_id,
                                house_id=house_id,
                                sub_house_id=sub_house_id,
                                raw_navigation_camera=nav_cam_video,
                                raw_manipulation_camera=manip_cam_video,
                                sensors_path=sensors_path,
                            )
                        )

                    

        random.seed(self.seed)
        random.shuffle(samples)
        return samples[: self.max_samples]

    def get_max_samples(self):
        # the max samples allowed for DDP is the min of samples read in all procs
        return min(
            [len(self.load_samples_for_proc_idx(proc_idx)) for proc_idx in range(self.num_procs)]
        )

    def partial_load_samples(self):
        return self.load_samples_for_proc_idx(self.proc_idx)

    def read_sensors(self, sensors_path, sub_house_id, additional_sensor_keys: List[str] = []):
        with h5py.File(sensors_path, "r") as f:
            grp = f[sub_house_id]
            sensor_keys = [
                "last_action_str",
                "initial_agent_location",
                "templated_task_spec",
            ] + additional_sensor_keys

            sensors = dict()
            task_dict = convert_byte_to_string(grp["templated_task_spec"][0], None)
            task_dict = json_templated_spec_to_dict(task_dict)

            for k in sensor_keys:
                if k == "initial_agent_location":
                    # pass
                    sensors[k] = grp["last_agent_location"][0]
                elif k == "last_action_str":
                    sensors[k] = [convert_byte_to_string(row, None) for row in grp[k]]
                elif k == "last_actions":
                    # last_actions are created from last_action_str which is always returned
                    continue
                elif k == "last_goal_agent_frame":
                    sensors[k] = [convert_byte_to_string(row, None) for row in grp[k]]
                elif k == "templated_task_spec":
                    sensors[k] = convert_byte_to_string(grp[k][0], None)
                elif k in [
                    "visible_target_4m_count",
                    "last_action_success",
                ]:
                    assert grp[k].shape[-1] == 1
                    sensors[k] = grp[k][:, 0]
                elif k in ["rooms_seen", "room_current_seen"]:
                    sensors[k] = grp[k][:-1]
                    sensors[f"{k}_output"] = grp[k][1:]
                elif k == "an_object_is_in_hand":
                    # TODO: check with kiana if this is ok
                    if k in grp:
                        # TODO KIANA REMOVE THIS AFTER DESCREPENCY IS RESOLVED
                        sensor_raw = grp[k][:]
                        if len(sensor_raw.shape) == 2:
                            sensors[k] = sensor_raw[:, 0]
                        else:
                            sensors[k] = sensor_raw
                    else:
                        sensors[k] = np.zeros(len(sensors["last_action_str"]))
                    assert np.all(sensors[k] <= 1)
                elif k == "relative_arm_location_metadata":
                    sensors[k] = grp[k][:]
                elif k in ["time_ids", "traj_index", "goal"]:
                    # time_ids, traj_index, and goal are created in the ilearn dataset
                    pass
                else:
                    raise NotImplementedError(f"Sensor {k} not implemented")
                    # sensors[k] = [convert_byte_to_string(row, None) for row in grp[k]]

        return sensors

    def load_video(self, video_path):

        # if "warped_front_camera" in video_path:
        #     video_path = video_path.replace("warped_front_camera", "raw_navigation_camera")
        # elif "warped_right_camera" in video_path:
        #     video_path = video_path.replace("warped_right_camera", "raw_manipulation_camera")
        return read_video(video_path, pts_unit="sec")[0]


class iLearnDataset(Dataset):
    def __init__(
        self,
        data_dir,
        subset,
        proc_idx=0,
        num_procs=1,
        max_samples=None,
        load_frames=True,
        sliding_window=None,
        input_sensors=("raw_navigation_camera",),
        output_sensors=[],
        reduce_action_redundancy=False,
        action_space=None,
    ):
        self.data_dir = data_dir
        self.reader = iLearnReader(data_dir, subset, proc_idx, num_procs, max_samples)
        self.load_frames = load_frames
        self.sliding_window = sliding_window
        self.samples = self.reader.partial_load_samples()
        self.visual_sensors = [x for x in input_sensors if is_a_visual_sensor(x)]
        self.non_visual_sensors = [x for x in input_sensors if not is_a_visual_sensor(x)]
        self.reduce_action_redundancy = reduce_action_redundancy
        self.add_boxes_to_img = False
        self.action_space = action_space
        self.output_sensors = output_sensors

        assert (
            not self.reduce_action_redundancy
        ) or subset == "train", "Reducing action redundancy is currently only supported for train."

        self.prob_sample_last_steps = 0

    def __len__(self):
        return len(self.samples)

    def set_prob_sample_last_steps(self, prob):
        self.prob_sample_last_steps = prob

    def select_window_slice(self, len_of_time_inds: int, start_idx=None, sliding_window=None):
        if sliding_window is None:
            sliding_window = self.sliding_window

        if sliding_window is None or len_of_time_inds <= sliding_window:
            return slice(0, len_of_time_inds)

        if start_idx is None:
            if random.random() < self.prob_sample_last_steps:
                start_idx = len_of_time_inds - sliding_window
            else:
                start_idx = random.randint(0, len_of_time_inds - sliding_window)

        return slice(start_idx, start_idx + sliding_window)

    def subsample_time_inds_to_reduce_action_redundancy(
        self,
        actions: np.ndarray,
        subsample_prob=3.0 / 4,
        action_subsample_factor=1.0 / 3,
        sliding_window=None,
    ):
        if sliding_window is None:
            sliding_window = self.sliding_window

        if sliding_window is None or len(actions) <= sliding_window:
            return slice(0, len(actions))

        if random.random() > subsample_prob:
            return self.select_window_slice(
                len_of_time_inds=len(actions), sliding_window=sliding_window
            )

        time_inds_partitioned_by_sequential_action_type = []
        last_action = None
        for time_ind, action in enumerate(actions):
            if last_action != action:
                time_inds_partitioned_by_sequential_action_type.append([])
                last_action = action

            time_inds_partitioned_by_sequential_action_type[-1].append(time_ind)

        candidate_time_inds_to_remove = sum(
            [part[1:] for part in time_inds_partitioned_by_sequential_action_type], []
        )
        random.shuffle(candidate_time_inds_to_remove)

        num_to_remove = int(
            np.random.binomial(
                len(candidate_time_inds_to_remove), p=1 - action_subsample_factor, size=1
            )[0]
        )
        # Always keep at least sliding_window number of actions
        num_to_remove = min(num_to_remove, len(actions) - sliding_window)

        time_inds_to_remove = set(candidate_time_inds_to_remove[:num_to_remove])

        new_time_inds = np.array(
            [time_ind for time_ind in range(len(actions)) if time_ind not in time_inds_to_remove]
        )

        return new_time_inds[
            self.select_window_slice(len(new_time_inds), sliding_window=sliding_window)
        ]

    def __getitem__(self, i):
        sample = deepcopy(self.samples[i])

        sensors = self.reader.read_sensors(
            sample["sensors_path"],
            sample["sub_house_id"],
            self.non_visual_sensors + self.output_sensors,
        )

        # For debugging
        # self.load_frames = True
        # self.sliding_window = 1000

        input_sensors = {}
        output_sensors = {}

        # We are reading this for tokenization purposes`

        original_actions_len = len(sensors["last_action_str"]) - 1


        # throw out the first action (null action)
        actions = sensors["last_action_str"][1:]

        select_indices = None

        if self.reduce_action_redundancy:
            select_indices = self.subsample_time_inds_to_reduce_action_redundancy(
                actions, sliding_window=self.sliding_window
            )
        else:
            select_indices = self.select_window_slice(original_actions_len)

        if select_indices is not None:
            time_ids = np.arange(original_actions_len)[select_indices]
            actions = np.array(actions)[select_indices]

        for sensor_name in self.non_visual_sensors:
            if sensor_name == "last_actions":
                input_sensors[sensor_name] = np.array(sensors["last_action_str"][:-1])[
                    select_indices
                ]
            elif sensor_name == "last_action_success":
                input_sensors[sensor_name] = sensors[sensor_name][:-1][select_indices]
            elif sensor_name in [
                "an_object_is_in_hand",
            ]:
                input_sensors[sensor_name] = sensors[sensor_name][:-1][select_indices]
            elif sensor_name == "goal":
                input_sensors[sensor_name] = json_templated_to_NL_spec(
                    sensors["templated_task_spec"]
                )
            elif sensor_name == "time_ids":
                if self.reader.subset == "train":
                    time_ids = time_ids + random.randint(0, 1000 - time_ids.shape[0])
                input_sensors[sensor_name] = time_ids
            elif sensor_name == "traj_index":
                input_sensors[sensor_name] = np.array([i] * len(actions))
            else:
                raise NotImplementedError(f"Sensor {sensor_name} not implemented")

        task_dict = json.loads(sensors["templated_task_spec"])
        if "Point" in task_dict["task_type"]:
            if task_dict["which_camera"] not in "_".join(self.visual_sensors):
                raise NotImplementedError(
                    f"Task {task_dict['task_type']} requires {task_dict['which_camera']} but only {self.visual_sensors} are available"
                )
        if self.load_frames:
            
            # throw out the last frame cause it doesn't have an action
            for sensor_name in self.visual_sensors:
                frames = self.reader.load_video(sample[sensor_name])[:-1]
                original_frames_len = len(frames)
                qual_control = video_quality_control(
                    sample["sample_id"],
                    sample[sensor_name],
                    original_frames_len,
                    original_actions_len,
                    frames,
                )
                if not qual_control:
                    return None

                frames = frames[select_indices]
                input_sensors[sensor_name] = frames
                
                # Load corresponding mask if it exists
                mask_key = f"{sensor_name}_mask"
                if mask_key in sample and os.path.exists(sample[mask_key]):
                    mask = np.array(Image.open(sample[mask_key]))
                    mask = (mask / 255).astype(np.uint8)
                    if mask is not None:
                        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)  # Add channel dimension
                        # mask_tensor = mask_tensor.repeat(len(frames), 1, 1, 1)
                        input_sensors[mask_key] = mask_tensor

        # KE: if you want to use last_goal agent as input you need to do more work!

        for sensor in self.output_sensors:
            if sensor == "last_goal_agent_frame":
                output_sensors["goal_agent_frame"] = sensors["last_goal_agent_frame"][1:][
                    select_indices
                ]
                output_sensors["goal_agent_frame"] = [
                    get_quantized_goal(goal) for goal in output_sensors["goal_agent_frame"]
                ]
            else:
                raise NotImplementedError(f"Sensor {sensor} not implemented")

        sample["input_sensors"] = dict(
            **input_sensors,
            **output_sensors,
            actions=actions,
            initial_agent_location=sensors["initial_agent_location"],
            templated_task_type=sensors["templated_task_spec"],
        )

        sample["prob_sample_last_steps"] = self.prob_sample_last_steps
        return sample


class iLearnMultitaskDataset(Dataset):
    def __init__(self, base_data_dir, dataset_names, pretrain=False, **kwargs):
        super().__init__()
        self.base_data_dir = base_data_dir
        self.dataset_names = dataset_names
        self.pretrain = pretrain
        dataset_cls = iLearnDataset
        if self.pretrain:
            raise NotImplementedError("Pretraining not implemented")
            dataset_cls = iLearnRandomDropMoveDataset
        self.datasets = []
        for name in self.dataset_names:
            print(f"Loading dataset: {name}")
            data_dir = os.path.join(base_data_dir, name)
            self.datasets.append(dataset_cls(data_dir, **kwargs))

        self.max_size = max(len(d) for d in self.datasets)

        # properties for last steps preference
        self.curr_prob_sample_last_steps = 0
        self.prob_decay_size = 0

    def __len__(self):
        return self.max_size * len(self.datasets)

    def set_prob_sample_last_steps(self, prob):
        for d in self.datasets:
            d.set_prob_sample_last_steps(prob)

    def init_prob_sample_last_steps(
        self, init_prob, final_prob, num_workers=4, num_gpu_per_node=1, num_node=1
    ):
        if num_gpu_per_node == 0:
            assert platform.system() == "Darwin"
            num_gpu_per_node = 1
        self.curr_prob_sample_last_steps = init_prob
        self.prob_decay_size = (init_prob - final_prob) / (
            len(self) / (num_workers * num_gpu_per_node * num_node)
        )

    def set_up_probabilty_to_sample_last_steps(self):
        self.curr_prob_sample_last_steps -= self.prob_decay_size
        self.set_prob_sample_last_steps(self.curr_prob_sample_last_steps)

    def __getitem__(self, index):
        try:  # TODO  Remove this after dataset generation is fixed
            # return sample in order D0[0],D1[0],D0[1],D1[1]
            # Choose the dataset based on the index
            dataset_index = index % len(self.datasets)
            dataset = self.datasets[dataset_index]
            # Choose the sample from the selected dataset
            sample_index = index // len(self.datasets)
            sample_index = sample_index % len(dataset)  # wrap around if out of range
            res = dataset[sample_index]
        except Exception as e:
            print("Exception in iLearnMultitaskDataset")
            print(e)
            print(traceback.format_exc())
            print("index", index)
            print("dataset_index", dataset_index)
            print("dataset", dataset.data_dir)
            res = None
        self.set_up_probabilty_to_sample_last_steps()
        return res
