import os
import platform
from typing import Callable, Dict, Any, Type

from omegaconf import OmegaConf

from data_generation.arg_parsers import parse_args_for_multi_task_mpqueue_manager
from data_generation.task_datagen_utils import (
    get_house_dataset,
    get_task_args_and_object_synsets,
    NUM_WORKERS_ON_SINGLE_DEVICE,
    construct_action_space_from_args,
)
from data_generation.manager_single_machine_mpqueue import manager_single_machine_mpqueue
from data_generation.path_planner_utils import REGISTERED_PLANNERS
from data_generation.path_planners import REGISTERED_PLANNERS
from environment.stretch_controller import StretchStochasticController, StretchController
from tasks import REGISTERED_TASK_SAMPLERS, REGISTERED_TASKS
from tasks.base_task_sampler import BaseTaskSampler
from utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
)
from utils.data_generation_utils.task_spec_sampling_utils import DEFAULT_PROB_RANDOMIZE_MATERIALS


def single_machine_mpqueue_entrypoint(
    save_dir: str,
    dataset_name: str,
    task_sampler_type: Type[BaseTaskSampler],
    device_to_task_sampler_kwargs: Callable[[int], Dict[str, Any]],
    house_repeats: int,
    split: str = "train",
):
    path_planner = REGISTERED_PLANNERS[task_sampler_type.task_type_str]()

    top_level_save_dir = os.path.join(
        save_dir,
        dataset_name,
    )

    manager_single_machine_mpqueue(
        nworkers=NUM_WORKERS_ON_SINGLE_DEVICE,
        house_repeats=house_repeats,
        split=split,
        top_level_save_dir=top_level_save_dir,
        constants=OmegaConf.create(),
        task_sampler_type=task_sampler_type,
        device_to_task_sampler_kwargs=device_to_task_sampler_kwargs,
        path_planner=path_planner,
    )


if __name__ == "__main__":
    args = parse_args_for_multi_task_mpqueue_manager()

    assert (
        args.task_type in REGISTERED_TASKS
    ), f"Unknown task type {args.task_type} for available {list(REGISTERED_TASKS.keys())}"

    action_space = construct_action_space_from_args(args.action_space_type, args.action_space_name)

    task_args, target_object_types, destination_object_types = get_task_args_and_object_synsets(
        task_type=args.task_type,
        max_steps=args.max_steps,
        action_space=action_space,
        closed_type=args.closed_type,
    )

    max_houses_per_split = dict(train=0, val=0, test=0)
    if platform.system() == "Darwin":
        max_houses_per_split[args.split] = 100
    else:
        max_houses_per_split[args.split] = 10_000_000
    dataset = get_house_dataset(
        house_dataset=args.house_dataset,
        debug=args.debug,
        max_houses_per_split=max_houses_per_split,
    )[args.split]

    task_sampler_args = dict(
        task_args=task_args,
        houses=dataset,
        house_inds=list(range(len(dataset))),
        target_object_types=target_object_types,
        controller_args=STRETCH_ENV_ARGS,
        max_tasks=float("inf"),
        task_type=REGISTERED_TASKS[args.task_type],
        controller_type=(
            StretchStochasticController if args.stochastic_controller else StretchController
        ),
        sample_per_house=1 * args.house_repeats,
        destination_object_types=destination_object_types,
        prob_randomize_materials=DEFAULT_PROB_RANDOMIZE_MATERIALS,
    )

    single_machine_mpqueue_entrypoint(
        save_dir=args.save_dir,
        dataset_name=args.dataset_name,
        task_sampler_type=REGISTERED_TASK_SAMPLERS[args.task_type],  # type:ignore
        device_to_task_sampler_kwargs=lambda device: {**task_sampler_args, "device": device},
        house_repeats=args.house_repeats,
        split=args.split,
    )
