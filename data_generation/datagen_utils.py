import pickle
import json
import os
import shutil
import string
import subprocess
import sys
import time
import traceback
from PIL import Image
from typing import Dict, Any, List, TYPE_CHECKING

import canonicaljson
import filelock
import h5py
import imageio
import numpy as np
import torch
import tqdm
from allenact.base_abstractions.sensor import SensorSuite
from beaker import Beaker
from omegaconf import OmegaConf

from environment.action_spaces import AbstractActionSpace
from utils.data_generation_utils.mp4_utils import save_frames_to_mp4
from vida_constants import ABS_PATH_OF_VIDA_DIR

try:
    from allenact.base_abstractions.sensor import SensorSuite
    from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
    from allenact.utils.misc_utils import all_equal
except ImportError:
    print(
        "AllenAct not installed, skipping imports from allenact, this may mean that some things will break."
    )

if TYPE_CHECKING:
    from data_generation.queues import Message

from data_generation.sensors import RawDepthSensorTHOR, RawRGBSensorTHOR


def format_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback_details = traceback.extract_tb(exc_traceback)

    for detail in reversed(traceback_details):
        if detail.filename.startswith(ABS_PATH_OF_VIDA_DIR):
            fname = detail.filename
            lineno = detail.lineno

            final_fname = traceback_details[-1].filename
            final_lineno = traceback_details[-1].lineno

            error_msg = (
                f'{exc_type.__name__}("{str(exc_value)}") was raised on line {lineno} of {fname}'
            )

            if fname != final_fname or lineno != final_lineno:
                error_msg += f" (originally thrown on {final_lineno} of {final_fname})"

            return error_msg

    return traceback.format_exc().replace("\n", "\\n")


def split_house_repeats_to_id(split: str, house: int, repeats: int):
    return f"split_{split}__house_{house:08d}__repeats_{repeats:02d}"


def recurse_write(group: h5py.Group, data: Dict[str, Any], should_keep):
    for k, v in data.items():
        if isinstance(v, Dict):
            recursively_write_to_hdf5(group=group.create_group(k), data=v)
        else:
            group.create_dataset(
                k,
                data=v.type(torch.float16 if v.dtype == torch.float32 else v.dtype)
                .cpu()
                .numpy()[should_keep, ...],
                compression="gzip",
                compression_opts=6,
            )


def skip_keys(obs, keys):
    obs_updated = {}
    for key in obs.keys():
        if key in keys:
            continue
        obs_updated[key] = obs[key]
    return obs_updated


def write_config_file(cfg: OmegaConf, top_level_save_dir: str, action_space: AbstractActionSpace):
    os.makedirs(top_level_save_dir, exist_ok=True)

    config_info_save_path = os.path.join(top_level_save_dir, "constants.yaml")
    action_space_save_path = os.path.join(top_level_save_dir, "action_space.pkl")

    with filelock.FileLock(os.path.join(top_level_save_dir, ".lock")):
        config_info = OmegaConf.to_container(cfg=cfg)
        cannonical_encoding = canonicaljson.encode_canonical_json(config_info).decode("utf-8")

        if not os.path.exists(config_info_save_path):
            with open(config_info_save_path, "w") as f:
                f.write(cannonical_encoding)

        if not os.path.exists(action_space_save_path):
            with open(action_space_save_path, "wb") as f:
                pickle.dump(action_space, f)

        with open(config_info_save_path, "r") as f:
            saved_info = f.read().strip()
            assert saved_info == cannonical_encoding.strip(), (
                "Config info does not match what was saved on disk. "
                " Please delete the file at"
                f" {config_info_save_path} and try again."
                " Saved config info:"
                f" {saved_info},"
                " current config info:"
                f" {cannonical_encoding}."
            )


def parse_queue_message(message: "Message", expected_split: str):
    id = message.body.strip()
    assert set(id) <= set(
        string.ascii_letters + string.digits + "_"
    ), f"Invalid ID '{id}', id must only contain a-Z, 0-9, and _"

    # Parse the ID into split, house, repeats
    run_info = {k: v for part in message.body.split("__") for k, v in [part.split("_")]}

    split = run_info["split"]
    assert expected_split == split
    house_index = int(run_info["house"])
    repeats = int(run_info["repeats"])
    assert len(run_info) == 3
    assert split in ["train", "train_unseen", "val", "test"]

    return {
        "id": id,
        "split": split,
        "house_index": int(house_index),
        "repeats": int(repeats),
    }


def recursively_write_to_hdf5(
    group: h5py.Group,
    data: Dict[str, Any],
    compression="gzip",
    compression_opts=6,
):
    for k, v in data.items():
        if isinstance(v, Dict):
            recursively_write_to_hdf5(
                group=group.create_group(k),
                data=v,
                compression=compression,
                compression_opts=compression_opts,
            )
        else:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            group.create_dataset(
                k,
                data=v,
                compression=compression,
                compression_opts=6,
            )




def save_trajectories(
    observations_list: List[Dict[str, Any]],
    sensor_suite: SensorSuite,
    save_dir: str,
    save_file_suffix: str = "",
    extra_obs_keys_to_save: List[str] = None,
    save_mp4s: bool = True,
):
    saved_paths = []
    try:
        sensor_uuids_for_hdf5 = []
        for sensor in sensor_suite.sensors.values():
            assert all(sensor.uuid in observations for observations in observations_list), (
                f"All observations must be present in the `observation_list`.\nSensors in sensor suite: "
                f" {[sensor.uuid for sensor in sensor_suite.sensors.values()]}."
                f"\nSensors in observation lists:"
                f"\n{[list(observations.keys()) for observations in observations_list]}."
            )

            if isinstance(sensor, RGBSensor):
                raise NotImplementedError(
                    "To save RGB videos please use the RawRGBSensorTHOR sensor rather than an RGBSensor"
                )

            elif isinstance(sensor, RawRGBSensorTHOR):
                if save_mp4s:
                    for i, observations in enumerate(observations_list):
                        rgb_save_path = os.path.join(
                            save_dir, f"{sensor.uuid}__{i}{save_file_suffix}.mp4"
                        )
                        saved_paths.append(rgb_save_path)

                        assert observations[sensor.uuid].dtype == torch.uint8
                        save_frames_to_mp4(
                            observations[sensor.uuid].cpu().numpy(),
                            file_path=rgb_save_path,
                            fps=10,
                        )
            elif isinstance(sensor, RawDepthSensorTHOR):
                if save_mp4s:
                    for i, observations in enumerate(observations_list):
                        depth_frames = observations[sensor.uuid].cpu().numpy()
                        assert depth_frames.dtype == np.uint8

                        depth_save_path = os.path.join(
                            save_dir, f"{sensor.uuid}__{i}{save_file_suffix}.mp4"
                        )
                        saved_paths.append(depth_save_path)

                        save_frames_to_mp4(
                            depth_frames,
                            file_path=depth_save_path,
                            fps=10,
                        )
            
            else:
                sensor_uuids_for_hdf5.append(sensor.uuid)

        sensor_uuids_for_hdf5.extend(extra_obs_keys_to_save or [])

        # for sensor_uuid in sensor_uuids_for_hdf5:
        hdf5_save_path = os.path.join(save_dir, f"hdf5_sensors{save_file_suffix}.hdf5")
        saved_paths.append(hdf5_save_path)
        with h5py.File(hdf5_save_path, "w") as hdf5_file:
            for i, observations in enumerate(observations_list):
                group = hdf5_file.create_group(f"{i}")
                recursively_write_to_hdf5(
                    group=group, data={k: observations[k] for k in sensor_uuids_for_hdf5}
                )
    except:
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)
        raise


def save_top_view_as_png(top_view: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    imageio.imwrite(file_path, top_view)


def find_all(name: str, path: str):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


def sha256sum_file(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def combine_beaker_datasets(
    beaker_obj: Beaker,
    combined_dataset_name: str,
    beaker_dataset_ids: List[str],
    tmp_save_dir: str,
    delete_old_datasets_on_beaker: bool,
    delete_local_data_on_exit: bool = True,
    allow_existing_downloads: bool = False,
):
    tmp_save_dir = os.path.join(tmp_save_dir, f".{combined_dataset_name}")
    assert allow_existing_downloads or not os.path.exists(tmp_save_dir)

    try:
        os.makedirs(tmp_save_dir, exist_ok=allow_existing_downloads)

        combined_dataset_dir = os.path.join(tmp_save_dir, "combined_dataset")
        os.makedirs(combined_dataset_dir, exist_ok=allow_existing_downloads)

        cfg_files = []
        for ds_id in beaker_dataset_ids:
            dataset_part_dir = os.path.join(tmp_save_dir, ds_id)

            if os.path.exists(dataset_part_dir):
                print(
                    f"Found existing dataset {ds_id} at {dataset_part_dir}. Skipping download but"
                    f" will still check that the files exist and are correct."
                )
                for f in tqdm.tqdm(beaker_obj.dataset.ls(ds_id)):
                    expected_file_path = os.path.join(dataset_part_dir, f.path)
                    if not os.path.exists(expected_file_path) or f.digest.value != sha256sum_file(
                        os.path.join(dataset_part_dir, f.path)
                    ):
                        delete_local_data_on_exit = False
                        raise RuntimeError(
                            f"Expected file '{expected_file_path}' to exist and have sha256sum '{f.digest.value}'"
                            f" but it does not. Will quit now but WILL NOT DELETE the downloaded datasets."
                        )
            else:
                download_failures = 0
                while True:
                    try:
                        print(f"Downloading dataset {ds_id}...")
                        print(
                            subprocess.check_output(
                                f"beaker dataset fetch {ds_id} -o {dataset_part_dir}".split()
                            ).decode()
                        )
                        break
                    except subprocess.CalledProcessError as e:
                        print(
                            f"Encountered error while fetching dataset!"
                            f" Printing the output returned up to this point:\n{e.output}."
                        )

                    download_failures += 1

                    if os.path.exists(dataset_part_dir):
                        shutil.rmtree(dataset_part_dir)

                    if download_failures > 5:
                        delete_local_data_on_exit = False
                        raise RuntimeError(
                            f"Have failed to download dataset '{ds_id}' 5 times."
                            f" Will quit now but WILL NOT DELETE the downloaded datasets."
                        )

                    time.sleep(30)  # Wait 30 seconds before trying again.

            beaker_obj.dataset.get(ds_id)

            found_cfg_files = find_all(name="config.json", path=dataset_part_dir)

            if len(found_cfg_files) == 0:
                continue

            assert len(found_cfg_files) == 1
            cfg_files.extend(found_cfg_files)

            # Saving the config file to the combined dataset
            cfg_file = found_cfg_files[0]
            cfg_save_path = os.path.join(
                combined_dataset_dir, os.path.relpath(cfg_file, start=dataset_part_dir)
            )
            if not os.path.exists(cfg_save_path):
                os.makedirs(os.path.dirname(cfg_save_path), exist_ok=True)
                shutil.copy(cfg_file, cfg_save_path)

            # beaker_obj.dataset.fetch(
            #     ds_id,
            #     target=output_path,
            #     quiet=False,
            #     validate_checksum=True,
            # )
        print("Done downloading")

        # Ensure all configs are the same.
        def read_json(path):
            with open(path, "r") as f:
                return json.load(f)

        cfg_files = find_all(name="config.json", path=tmp_save_dir)
        cfg_jsons = [read_json(p) for p in cfg_files]
        assert all_equal(cfg_jsons)

        for ds_id in beaker_dataset_ids:
            dataset_dir = os.path.join(tmp_save_dir, ds_id)

            for p in find_all(name="traj.hdf5", path=dataset_dir):
                old_dir = os.path.dirname(p)
                new_dir = os.path.join(
                    combined_dataset_dir, os.path.relpath(old_dir, start=dataset_dir)
                )

                if (
                    os.path.isdir(old_dir)
                    and os.path.exists(os.path.join(old_dir, "success.txt"))
                    and not os.path.exists(new_dir)
                ):
                    os.makedirs(os.path.dirname(new_dir), exist_ok=True)
                    shutil.move(old_dir, new_dir)

        print("Uploading combined dataset to beaker...")
        saved_ds = beaker_obj.dataset.create(combined_dataset_name, combined_dataset_dir)
        print(f"Combined dataset saved with ID {saved_ds.id}")

        if delete_old_datasets_on_beaker:
            print(f"Now deleting old datasets on Beaker")
            for ds_id in beaker_dataset_ids:
                beaker_obj.dataset.delete(ds_id)
    finally:
        if delete_local_data_on_exit and os.path.exists(tmp_save_dir):
            shutil.rmtree(tmp_save_dir)

    return saved_ds



def print_error(*args, **kwargs):
    if "flush" not in kwargs:
        kwargs["flush"] = True
    print(*args, file=sys.stderr, **kwargs)
