import queue
import time
from typing import Union, Dict, Any, Type, Callable

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from setproctitle import setproctitle as ptitle

from allenact.base_abstractions.task import TaskSampler
from allenact.utils.system import get_logger, init_logging
from data_generation.datagen_utils import (
    split_house_repeats_to_id,
)
from data_generation.path_planners import PathPlanner
from data_generation.queues import FromToQueue
from data_generation.worker import offline_dataset_worker

mp = mp.get_context("forkserver") if torch.cuda.is_available() else mp.get_context("spawn")


def manager_single_machine_mpqueue(
    nworkers: int,
    house_repeats: int,
    split: str,
    top_level_save_dir: str,
    constants: OmegaConf,
    task_sampler_type: Type[TaskSampler],
    device_to_task_sampler_kwargs: Callable[[Union[int]], Dict[str, Any]],
    path_planner: PathPlanner,
):
    ptitle("Offline Expert Dataset Manager")

    init_logging()

    to_worker_queue = mp.Queue()
    from_worker_queue = mp.Queue()

    devices = [None] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))

    task_sampler_kwargs_per_device = []
    for d in devices:
        task_sampler_kwargs_per_device.append(device_to_task_sampler_kwargs(d))

    house_inds = sorted(
        list(set(sum((tsk["house_inds"] for tsk in task_sampler_kwargs_per_device), [])))
    )

    ids = []
    for i in house_inds:
        ids.append(split_house_repeats_to_id(split=split, house=i, repeats=house_repeats))
    get_logger().info(f"Generated {len(ids)} job ids for workers to process")
    ids.reverse()

    # 32767 is the max number of items that can be put in an mp.Queue
    # we'll start by putting that many items in the queue and add more later below
    for _ in range(32767):
        if len(ids) == 0:
            break
        to_worker_queue.put(ids.pop())

    processes = []
    finished_ids = []
    try:
        for i in range(nworkers):
            p = mp.Process(
                target=offline_dataset_worker,
                kwargs=dict(
                    constants=constants,
                    task_sampler_type=task_sampler_type,
                    task_sampler_kwargs=task_sampler_kwargs_per_device[i % len(devices)],
                    expected_split=split,
                    worker_ind=i,
                    lor_queue=FromToQueue(
                        from_queue=to_worker_queue,
                        to_queue=from_worker_queue,
                    ),
                    device=devices[i % len(devices)],
                    path_planner=path_planner,
                    top_level_save_dir=top_level_save_dir,
                    metrics_json_dir="experiment_output/",
                ),
            )
            p.start()
            time.sleep(1)
            processes.append(p)

        last_done = time.time()
        last_ping = time.time()
        while len(finished_ids) < len(house_inds):
            try:
                finished_id = from_worker_queue.get(timeout=0.1)
                get_logger().info(f"{finished_id} finished processing.")
                finished_ids.append(finished_id)
                last_done = time.time()
                last_ping = last_done

                if len(ids) > 0:
                    to_worker_queue.put(ids.pop())
            except queue.Empty:
                time.sleep(30)
                if time.time() - last_ping > 60:
                    get_logger().info("Manager: no new results in 60 seconds.")
                    last_ping = time.time()

            if time.time() - last_done > 60 * 60 * 1:
                raise TimeoutError
    finally:
        get_logger().info(
            f"Dataset manager quitting ({len(finished_ids)}/{len(task_sampler_kwargs_per_device[0]['houses'])}"
            f" scenes complete) ..."
        )
        for p in processes:
            p.join(1)
