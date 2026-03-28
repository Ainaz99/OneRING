import glob
import json
import multiprocessing as mp
import os
import platform
import queue
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Union, Dict, Any, Type, Optional

import numpy as np
import torch
from allenact.base_abstractions.sensor import SensorSuite
from allenact.utils.misc_utils import md5_hash_str_as_int
from allenact.utils.tensor_utils import batch_observations, to_device_recursively
from filelock import FileLock
from omegaconf import OmegaConf
from setproctitle import setproctitle as ptitle

from data_generation.datagen_utils import (
    write_config_file,
    parse_queue_message,
    save_trajectories,
    format_exception,
    print_error,
)
from data_generation.path_planners import PathPlanner
from data_generation.queues import LocalOrRemoteQueue, Message
from tasks import BaseTaskSampler
from utils.data_generation_utils.exception_utils import (
    PlannerFailedException,
    HouseInvalidForTaskException,
    TaskSamplerInInvalidStateError,
    IrrecoverablePlannerFailureDueToHouseException,
    TaskDifficultyIncorrectException,
)
from utils.data_generation_utils.mp4_utils import save_frames_to_mp4


def increment_metrics_json(
    metrics_json_dir: Optional[str],
    success: int = 0,
    failure: int = 0,
    crashed: int = 0,
    already_processed: int = 0,
    worker_ind: int = 0,
    planner_failures: int = 0,
    planner_successes: int = 0,
):
    if metrics_json_dir is None:
        return

    os.makedirs(metrics_json_dir, exist_ok=True)
    path = os.path.join(metrics_json_dir, "metrics.json")
    with FileLock(path + ".lock"):
        if os.path.exists(path):
            with open(path, "r") as f:
                metrics = json.load(f)
                if "already_processed" not in metrics:
                    metrics["already_processed"] = 0
                if "planner_failures" not in metrics:
                    metrics["planner_failures"] = 0
                if "planner_successes" not in metrics:
                    metrics["planner_successes"] = 0
        else:
            metrics = dict(
                success=0,
                failure=0,
                crashed=0,
                already_processed=0,
                planner_failures=0,
                planner_successes=0,
            )

        metrics["success"] += success
        metrics["failure"] += failure
        metrics["crashed"] += crashed
        metrics["already_processed"] += already_processed
        metrics["planner_successes"] += planner_successes
        metrics["planner_failures"] += planner_failures

        with open(path, "w") as f:
            json.dump(metrics, f)


def save_logging_info(
    task_type_str: str,
    logging_save_dir: Optional[str],
    time_taken: float,
    split: str,
    house_index: int,
    repeats: int,
    num_saved_trajectories: int,
    retries: int,
):
    if logging_save_dir is None:
        return

    keys = [
        "task_type_str",
        "time_taken",
        "split",
        "house_index",
        "repeats",
        "num_saved_trajectories",
        "retries",
    ]

    os.makedirs(logging_save_dir, exist_ok=True)
    path = os.path.join(logging_save_dir, "logs.tsv")
    with FileLock(path + ".lock"):
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("\t".join(keys) + "\n")

        def to_str(inp):
            if isinstance(inp, float):
                return f"{inp:.1f}"
            else:
                return str(inp)

        with open(path, "a") as f:
            kwargs = locals()
            f.write("\t".join(to_str(kwargs[k]) for k in keys) + "\n")



def offline_dataset_worker(
    constants: OmegaConf,
    task_sampler_type: Type[BaseTaskSampler],
    task_sampler_kwargs: Dict[str, Any],
    path_planner: PathPlanner,
    worker_ind: int,
    lor_queue: LocalOrRemoteQueue,
    device: Optional[Union[torch.device, int]],
    expected_split: str,
    top_level_save_dir: str,
    metrics_json_dir: Optional[str],
    is_alive_queue: Optional[mp.Queue] = None,
):
    ptitle(f"Offline Dataset Worker {worker_ind}")
    print(f"Starting worker {worker_ind}", flush=True)

    if device is None:
        device = -1

    action_space = task_sampler_kwargs["task_args"]["action_space"]

    write_config_file(
        cfg=constants, top_level_save_dir=top_level_save_dir, action_space=action_space
    )
    task_sampler: Optional[BaseTaskSampler] = (
        None  # Initialized only when needed below, faster when running scripts locally
    )
    default_message_timeout = lor_queue.message_timeout

    message: Optional[Message] = None

    # A sequential irrecoverable failures is a failure where we must give up generating additional data from a house
    # because either the tasks cannot be sampled or because the planner simply cannot seem to generate successful plans.
    # If we see many sequential such failures, it's generally a sign that something has gone every wrong (e.g. the
    # AI2-THOR process or task sampler has gotten into a bad state). In such a case, we simply kill the worker.
    # TODO: Restart the worker instead.
    num_sequential_irrecoverable_failures = 0
    max_allowed_sequential_irrecoverable_failures = 5

    # The number of sequential task-sampler / planner failures we allow before its considered an irrecoverable failure
    max_allowed_sequential_task_sampler_failures = 10
    max_allowed_sequential_planner_failures = 10
    try:
        while True:
            message = lor_queue.get(timeout=2)
            house_processing_start_time = time.time()
            last_message_refresh_time = time.time()
            message_timeout = default_message_timeout

            trajectories_to_generate_info = parse_queue_message(
                message, expected_split=expected_split
            )

            split = trajectories_to_generate_info["split"]
            house_index = trajectories_to_generate_info["house_index"]
            repeats = trajectories_to_generate_info["repeats"]

            def save_logging_info_for_house(num_saved_trajectories: int, retries: int):
                save_logging_info(
                    task_type_str=task_sampler_type.task_type_str,
                    logging_save_dir=metrics_json_dir,
                    time_taken=time.time() - house_processing_start_time,
                    split=split,
                    house_index=house_index,
                    repeats=repeats,
                    num_saved_trajectories=num_saved_trajectories,
                    retries=retries,
                )

            split_save_dir = os.path.join(top_level_save_dir, split)
            os.makedirs(split_save_dir, exist_ok=True)

            traj_id_without_repeat = f"split_{split}__house_{house_index:06d}"

            cur_save_dir = os.path.join(split_save_dir, f"{house_index:06d}")

            success_save_path = os.path.join(cur_save_dir, "success.txt")

            if os.path.exists(success_save_path):
                print(
                    f"[Worker {worker_ind}] skipping {traj_id_without_repeat} as it was already processed.",
                    flush=True,
                )
                increment_metrics_json(
                    metrics_json_dir=metrics_json_dir, already_processed=1, worker_ind=worker_ind
                )
                lor_queue.mark_complete(message)
                continue

            if os.path.exists(cur_save_dir):
                # If the data wasn't saved successfully previously, delete what we have so far
                print(
                    f"[Worker {worker_ind}] {cur_save_dir} ALREADY EXISTS. Attempting to delete.",
                    flush=True,
                )
                try:
                    # NOTE: this should always just work, but there seems to be a
                    # rare edge case that will kill the worker in question
                    shutil.rmtree(cur_save_dir)
                except OSError as e:
                    print_error(
                        f"[Worker {worker_ind}] Could not delete {cur_save_dir} using shutil.rmtree(...)."
                        f" Trying to recover from this error. Error: {format_exception()}.",
                    )
                    if e.errno == 39:
                        print_error(
                            f"[Worker {worker_ind}] Encountered a directory not empty error for {cur_save_dir}."
                            f" Attempting to delete contents individually.",
                        )
                        paths_to_delete = os.listdir(cur_save_dir)
                        if len(paths_to_delete) == 0:
                            print_error(
                                f"[Worker {worker_ind}] {cur_save_dir} is empty but encountered a directory not"
                                f" empty error. Don't know how to handle this, skipping..."
                            )
                            increment_metrics_json(
                                metrics_json_dir=metrics_json_dir, failure=1, worker_ind=worker_ind
                            )
                            lor_queue.mark_complete(message)
                            continue

                        for file_name in paths_to_delete:
                            file_path = os.path.join(success_save_path, file_name)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

                        # Try to delete the directory again
                        try:
                            shutil.rmtree(cur_save_dir)
                        except:
                            print_error(
                                f"[Worker {worker_ind}] Could not delete {cur_save_dir} using shutil.rmtree(...)"
                                f"after deleting contents individually. Don't know how to handle this, skipping...",
                            )
                            increment_metrics_json(
                                metrics_json_dir=metrics_json_dir, failure=1, worker_ind=worker_ind
                            )
                            lor_queue.mark_complete(message)
                            continue
                    else:
                        print_error(
                            f"[Worker {worker_ind}] Unable to delete or handle an incomplete"
                            f" directory at {cur_save_dir}. SKIPPING THIS HOUSE.",
                        )
                        increment_metrics_json(
                            metrics_json_dir=metrics_json_dir, failure=1, worker_ind=worker_ind
                        )
                        lor_queue.mark_complete(message)
                        continue

            observations_per_repeat = []
            sensor_suite: Optional[SensorSuite] = None
            irrecoverable_failure_in_house = False

            num_sequential_task_sampler_failures = 0
            num_sequential_planner_failures = 0
            retry = 0
            for retry in range(repeats * max_allowed_sequential_planner_failures):
                # Tell the main process that we are still alive
                if is_alive_queue is not None:
                    is_alive_queue.put(True)

                # # SQS queue messages can time out after a certain number of minutes resulting in them
                # # being put back onto the queue for other workers to grab. Since we are doing many
                # # repeats/retries with a single message this might result in us processing the same
                # # message multiple times. The following few lines will refresh a message timeout
                # # every so often to attempt to ensure this doesn't happen.
                # if time.time() - last_message_refresh_time > 0.75 * default_message_timeout:
                #     last_message_refresh_time = time.time()
                #     message_timeout += default_message_timeout
                #     lor_queue.refresh_message(message=message, new_timeout=message_timeout)

                repeat = len(observations_per_repeat)
                if repeat >= repeats:
                    retry -= (
                        1  # We don't want to count this as a retry when we log this counter below
                    )
                    break

                if (
                    num_sequential_task_sampler_failures
                    >= max_allowed_sequential_task_sampler_failures
                ):
                    print_error(
                        f"[Worker {worker_ind}] encountered an error when calling next calling next_task for"
                        f" {traj_id_without_repeat} {num_sequential_task_sampler_failures} consecutive times."
                        f" This is unrecoverable, NO ADDITIONAL DATA WILL BE GENERATED FOR THIS HOUSE."
                        f" Traceback:\n {format_exception()}"
                    )
                    irrecoverable_failure_in_house = True
                    num_sequential_irrecoverable_failures += 1
                    retry -= (
                        1  # We don't want to count this as a retry when we log this counter below
                    )
                    break

                if num_sequential_planner_failures >= max_allowed_sequential_planner_failures:
                    print(
                        f"[Worker {worker_ind}] Path planner failed for {traj_id_without_repeat} on repeat {repeat} across"
                        f" {num_sequential_planner_failures} retries."
                        f" This is irrecoverable, NO ADDITIONAL DATA WILL BE GENERATED FOR THIS HOUSE.",
                        flush=True,
                    )
                    irrecoverable_failure_in_house = True
                    num_sequential_irrecoverable_failures += 1
                    retry -= (
                        1  # We don't want to count this as a retry when we log this counter below
                    )
                    break

                if task_sampler is None:
                    task_sampler = task_sampler_type(**task_sampler_kwargs)

                traj_id = f"task__{task_sampler.task_type_str.lower()}__{traj_id_without_repeat}_repeat_{repeat:02d}_retry_{retry:02d}"

                seed = (md5_hash_str_as_int(traj_id)) % (2**30)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                task_sampler.set_seed(seed)
                try:
                    task = task_sampler.next_task(force_advance_scene=True, house_index=house_index)
                    num_sequential_task_sampler_failures = 0
                except HouseInvalidForTaskException:
                    # This house cannot generate any tasks because it simply doesn't work for the task sampler
                    # this means we should skip it entirely.
                    print_error(
                        f"[Worker {worker_ind}] encountered a HouseInvalidForTaskException when calling next_task for"
                        f" {traj_id} {num_sequential_task_sampler_failures}. This is unrecoverable,"
                        f" NO DATA WILL BE GENERATED FOR THIS HOUSE. Traceback: {format_exception()}"
                    )
                    irrecoverable_failure_in_house = True
                    # We DO NOT do num_sequential_irrecoverable_failures += 1 here because we don't want to
                    # count this as an irrecoverable failure for the house. These types of exceptions can just happen.
                    break
                except TaskSamplerInInvalidStateError:
                    raise
                except:
                    print_error(
                        f"[Worker {worker_ind}] encountered an error when calling next_task for"
                        f" {traj_id} {num_sequential_task_sampler_failures} consecutive times."
                        f" Traceback: {format_exception()}"
                    )
                    num_sequential_task_sampler_failures += 1
                    continue

                if sensor_suite is None:
                    sensor_suite = task.sensor_suite

                print(
                    f"[Worker {worker_ind}] starting {traj_id} ({task_sampler.task_type_str}) with instruction: {task.to_string()}",
                    flush=True,
                )
                try:
                    if path_planner.is_planner_guaranteed_to_fail(task):
                        raise PlannerFailedException("Planner is guaranteed to fail.")

                    observations_list = path_planner.plan(task)
                    num_sequential_planner_failures = 0
                    print(
                        f"[Worker {worker_ind}] completed planning for house {house_index} and repeat {repeat}.",
                        flush=True,
                    )
                    increment_metrics_json(
                        metrics_json_dir=metrics_json_dir,
                        planner_successes=1,
                        worker_ind=worker_ind,
                    )
                except IrrecoverablePlannerFailureDueToHouseException:
                    # Similarly as for a HouseInvalidForTaskException, the planner does not think this
                    # house is suitable for this task. We should skip it entirely.
                    print_error(
                        f"[Worker {worker_ind}] encountered a IrrecoverablePlannerFailureDueToHouseException when calling plan for"
                        f" {traj_id} {num_sequential_task_sampler_failures}. This is unrecoverable,"
                        f" NO ADDITIONAL DATA WILL BE GENERATED FOR THIS HOUSE. Traceback: {format_exception()}"
                    )
                    irrecoverable_failure_in_house = True
                    # We DO NOT do num_sequential_irrecoverable_failures += 1 here because we don't want to
                    # count this towards needing the kill this worker. These types of exceptions can just happen.
                    break
                except (PlannerFailedException, AssertionError) as e:
                    # # TODO KIANA comment out
                    if platform.system() == "Darwin" and "Task is trivial" not in str(e) and len(task.get_observation_history()):
                        failed_directory = os.path.join("experiment_output", "failed_traj_debug")
                        os.makedirs(failed_directory, exist_ok=True)
                        video_id = os.path.join(
                            failed_directory, f"{house_index:06d}_{repeat:02d}_{retry:02d}_{str(e)}"
                        )
                        raw_navigation_camera = np.array(
                            [x["raw_navigation_camera"] for x in task.get_observation_history()]
                        )
                        save_frames_to_mp4(
                            raw_navigation_camera,
                            file_path=f"{video_id}_rgb.mp4",
                            fps=10,
                        )
                        camera_follow_agent = np.array(
                            [x["camera_follow_agent"] for x in task.get_observation_history()]
                        )
                        save_frames_to_mp4(
                            camera_follow_agent,
                            file_path=f"{video_id}_follow_agent.mp4",
                            fps=10,
                        )
                        with open(f"{video_id}.txt", "w") as f:
                            f.write("Target Objects:\n")
                            f.write(str(task.task_info["synset_to_object_ids"]))
                            f.write("\n")
                            f.write("ERROR IS:\n")
                            f.write(str(e))
                            f.write("\n")
                            f.write(str(task.task_info))
                            f.write("\n")
                            f.write(str(task.controller.agent_params))
                    # Assertion error catching added for the case where the task hit `max_steps`.
                    if (
                        isinstance(e, AssertionError)
                        and "assert not self.is_done()" not in traceback.format_exc()
                    ):
                        raise

                    if not isinstance(e, TaskDifficultyIncorrectException):
                        increment_metrics_json(
                            metrics_json_dir=metrics_json_dir,
                            planner_failures=1,
                            worker_ind=worker_ind,
                        )

                    num_sequential_planner_failures += 1

                    print_error(
                        f"[Worker {worker_ind}] Failed {traj_id} planning, retrying... Error: {format_exception()}"
                    )
                    continue
                except TimeoutError:
                    print_error(
                        f"[Worker {worker_ind}] TimeoutError for {traj_id}. This suggests that AI2-THOR has died."
                        f" Will mark this trajectory as irrecoverable, attempt to kill any existing AI2-THOR process"
                        f" and restart AI@-THOR."
                    )
                    try:
                        task_sampler.controller.stop()
                    except:
                        pass

                    task_sampler = (
                        None  # Should cause the task sampler to be restarted on the next iteration
                    )
                    irrecoverable_failure_in_house = True
                    break

                # TODO: A bit of a hack to deal with differently sized task_info strings.
                for obs in observations_list:
                    if "templated_task_spec" in obs:
                        obs["templated_task_spec"] = observations_list[-1]["templated_task_spec"]

                observations_dict = batch_observations(
                    observations_list, device=(device if device != -1 else None)
                )
                if device != -1:
                    # Otherwise can lead to CUDA OOM errors
                    to_device_recursively(observations_dict, device=torch.device("cpu"))
                observations_per_repeat.append(observations_dict)

                # Everything was successful, reset the number of sequential irrecoverable failures!
                num_sequential_irrecoverable_failures = 0

            if irrecoverable_failure_in_house:
                lor_queue.mark_complete(message)
                print(
                    f"[Worker {worker_ind}] irrecoverable failure for house {house_index}.",
                    flush=True,
                )

                if (
                    num_sequential_irrecoverable_failures
                    >= max_allowed_sequential_irrecoverable_failures
                ):
                    # Can only happen if everything fails multiple times in a row, this suggests that something is
                    # really wrong so we quit the worker
                    print_error(
                        f"[Worker {worker_ind}] too many sequential task sampler failures, exiting..."
                    )
                    save_logging_info_for_house(
                        num_saved_trajectories=0,
                        retries=retry + 1,
                    )
                    increment_metrics_json(
                        metrics_json_dir=metrics_json_dir,
                        failure=1,
                        crashed=1,
                        worker_ind=worker_ind,
                    )
                    sys.exit(1)

            try:
                if len(observations_per_repeat) == 0:
                    # Only a failure if we have NO trajectories for this house.
                    save_logging_info_for_house(
                        num_saved_trajectories=0,
                        retries=retry + 1,
                    )
                    increment_metrics_json(
                        metrics_json_dir=metrics_json_dir, failure=1, worker_ind=worker_ind
                    )
                    continue

                if os.path.exists(cur_save_dir):
                    if (
                        len(glob.glob(os.path.join(cur_save_dir, "*.mp4"))) == 0
                        and len(glob.glob(os.path.join(cur_save_dir, "*.hdf5"))) == 0
                    ):
                        print_error(
                            f"[Worker {worker_ind}] Found existing files in the save directory {cur_save_dir}, this should not happen!"
                        )
                        increment_metrics_json(
                            metrics_json_dir=metrics_json_dir, failure=1, worker_ind=worker_ind
                        )
                        continue

                save_trajectories(
                    observations_list=observations_per_repeat,
                    sensor_suite=sensor_suite,
                    save_dir=cur_save_dir,
                    save_file_suffix="",  # TODO: Do we want a suffix in the future?
                )
                Path(success_save_path).touch()
                save_logging_info_for_house(
                    num_saved_trajectories=len(observations_per_repeat),
                    retries=retry + 1,
                )
                increment_metrics_json(
                    metrics_json_dir=metrics_json_dir, success=1, worker_ind=worker_ind
                )
            except:
                increment_metrics_json(
                    metrics_json_dir=metrics_json_dir, failure=1, worker_ind=worker_ind
                )
                print_error(
                    f"[Worker {worker_ind}] Failed to save trajectories for house {house_index}."
                    f" Error: {format_exception()}"
                )

                if os.path.exists(success_save_path):
                    os.remove(success_save_path)

                if os.path.exists(cur_save_dir):
                    os.rmdir(cur_save_dir)

                msg = traceback.format_exc()
                if "CUDA" in msg or "out of memory" in msg:
                    print_error(
                        f"[Worker {worker_ind}] Above error appears to be a CUDA or memory related error,"
                        f" we cannot recover from this. Exiting."
                    )
                    raise
            finally:
                lor_queue.mark_complete(message)

            print(
                f"[Worker {worker_ind}] completed house {house_index} with {repeats} repeats.",
                flush=True,
            )

    except queue.Empty:
        print(f"[Worker {worker_ind}] finished.", flush=True)
    except Exception:
        print_error(
            f"[Worker {worker_ind}] encountered an exception."
            f" Last message id {message.message_id if message is not None else None}."
            f" Exception: \n{traceback.format_exc()}"
        )
        raise
    finally:
        print(f"[Worker {worker_ind}] exiting...", flush=True)
        try:
            task_sampler.close()
        except:
            pass
