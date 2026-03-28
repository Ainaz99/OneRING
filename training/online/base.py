import abc
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, Union

import os
import wandb
import torch
import torch.nn as nn
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph, Preprocessor
from allenact.base_abstractions.sensor import ExpertActionSensor, Sensor, SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins

from environment.action_spaces import SPOCV1ActionSpace
from environment.stretch_controller import StretchController
from tasks.multi_task_eval_sampler import MultiTaskSampler
from tasks.task_specs import TaskSpecDatasetList, TaskSpecSamplerInfiniteList
from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from utils.data_utils import Hdf5TaskSpecs, LazyJsonHouses, LazyJsonTaskSpecs
from utils.task_sampler_utils import TaskSpecPartitioner
from utils.type_utils import AbstractTaskArgs
from vida_constants import WANDB_DEFAULT_PROJECT_NAME


def task_sampler_args_builder(
    mode: Literal["train", "val", "test"],
    task_specs: Union[LazyJsonTaskSpecs, Hdf5TaskSpecs],
    houses: LazyJsonHouses,
    on_server: bool,
    sensors: List[Sensor],
    action_space: SPOCV1ActionSpace,
    max_steps: int,
    process_ind: int = 0,
    total_processes: int = 1,
    controller_args: Dict[str, Any] = STRETCH_ENV_ARGS,
    controller_type: Type = StretchController,
    prob_randomize_materials: float = 0,
    deterministic_cudnn: bool = False,
    reward_configs: Optional[Dict[str, Any]] = None,
    devices: Optional[List[int]] = None,
    max_houses: Optional[int] = 100000,
):
    assert on_server or max_houses is not None, (
        "max_houses must be provided if not on server. "
        f"Currently max_houses={max_houses} and on_server={on_server}"
    )

    if isinstance(task_specs, LazyJsonTaskSpecs):
        # get task specs and houses for the current process
        partitioner = TaskSpecPartitioner(
            task_specs=task_specs,
            houses=houses,
            process_ind=process_ind,
            total_processes=total_processes,
            max_houses=max_houses,
        )
        selected_task_specs = partitioner.task_specs_for_curr_process
        selected_houses = partitioner.houses_for_curr_process
        selected_house_inds = partitioner.house_inds_for_curr_process
    elif isinstance(task_specs, Hdf5TaskSpecs):
        assert (
            task_specs.proc_id == process_ind
        ), f"Hdf5TaskSpecs.proc_id ({task_specs.proc_id}) must match process_ind ({process_ind})"
        assert (
            task_specs.total_procs == total_processes
        ), f"Hdf5TaskSpecs.total_procs ({task_specs.total_procs}) must match total_processes ({total_processes})"
        selected_task_specs = task_specs
        selected_house_inds = [task_spec["house_index"] for task_spec in selected_task_specs]
        selected_houses = houses.select(selected_house_inds)
    else:
        raise NotImplementedError(
            f"task_specs must be LazyJsonTaskSpecs or Hdf5TaskSpecs not {type(task_specs)}"
        )

    # create task_spec sampler
    if mode == "train":
        house_index_to_task_specs = defaultdict(lambda: [])
        for task_spec in selected_task_specs:
            house_index_to_task_specs[task_spec["house_index"]].append(task_spec)

        task_spec_sampler = TaskSpecSamplerInfiniteList(
            house_index_to_task_specs, shuffle=True, repeat_house_until_forced=True
        )
    else:
        task_spec_sampler = TaskSpecDatasetList(selected_task_specs)

    # select device
    if on_server:
        device = devices[process_ind % len(devices)]
    else:
        device = None

    # create AbstratTaskArgs
    task_args = AbstractTaskArgs(
        action_space=action_space,
        sensors=sensors,
        max_steps=max_steps,
    )

    return {
        "mode": mode,
        "task_args": task_args,
        "houses": selected_houses,
        "house_inds": selected_house_inds,
        "task_spec_sampler": task_spec_sampler,
        "controller_args": controller_args,
        "controller_type": controller_type,
        "device": device,
        "visualize": False,
        "always_allocate_a_new_stretch_controller_when_reset": True,
        "retain_agent_pose": False,
        "prob_randomize_materials": prob_randomize_materials,
    }


@dataclass
class BaseConfigParams:
    num_train_processes: int = 2
    distributed_nodes: int = 1
    test_on_validation: bool = True
    dataset_dirs: List[str] = field(
        default_factory=lambda: [
            "/net/nfs/prior/datasets/spocv2_datasets/PickupPointOnObject",
            "/net/nfs/prior/datasets/spocv2_datasets/GoNearPointOnObject",
            "/net/nfs/prior/datasets/spocv2_datasets/GoToPointOnFloor",
        ]
    )
    use_jsonl_to_load_task_specs: bool = False
    max_steps: int = 500
    max_houses: Optional[int] = None
    max_task_specs: Optional[int] = None
    tag: str = "ObjectNavType-RL"
    actions: SPOCV1ActionSpace = SPOCV1ActionSpace()
    resume_allenact_ckpt: bool = False
    load_RL_ckpt: bool = False


class BaseConfig(ExperimentConfig, ABC):
    def __init__(
        self,
        params: "BaseConfigParams",
    ):
        super().__init__()
        self.params = params
        self._preproc: Sequence[Preprocessor] = []
        self.num_validation_processes = 1  # allenact only supports 1 validation process
        self.num_test_processes = 2 * torch.cuda.device_count() if torch.cuda.is_available() else 1

        self.houses = dict(
            train=self.get_houses(subset="train"),
            val=self.get_houses(subset="val"),
        )

    def tag(self) -> str:
        return self.params.tag

    @abc.abstractmethod
    def preprocessors(self):
        raise NotImplementedError

    @abc.abstractmethod
    def create_model(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def download_ckpt(
        self,
        training_run_id: str,
        ckpt: str,
        output_basedir: str,
        wandb_source_project: str = WANDB_DEFAULT_PROJECT_NAME,
    ) -> str:
        api = wandb.Api()
        run = api.run(f"prior-ai2/{wandb_source_project}/{training_run_id}")

        training_run_name = run.config["exp_name"]
        exp_base_dir = os.path.join(output_basedir, training_run_name)
        ckpt_dir = os.path.join(exp_base_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)

        if ckpt is None:
            raise Exception("ckptStep is None")
        else:
            if os.path.isfile(os.path.join(ckpt_dir, "model.ckpt")):
                ckpt_pth = os.path.join(ckpt_dir, "model.ckpt")
                print("-" * 80)
                print(
                    "ckpt has been downloaded before. Use the old one",
                    ckpt_pth,
                )
                print("-" * 80)
            else:
                ckpt_fn = f"prior-ai2/{wandb_source_project}/ckpt-{training_run_id}-{ckpt}:latest"
                artifact = api.artifact(ckpt_fn)
                artifact.download(ckpt_dir)
                ckpt_pth = os.path.join(ckpt_dir, "model.ckpt")

        if ckpt_pth is None:
            raise Exception("weight was not found", ckpt_fn)
        return ckpt_pth
    

    def get_houses(self, subset) -> LazyJsonHouses:
        return LazyJsonHouses.from_dir(
            OBJAVERSE_HOUSES_DIR,
            subset=subset,
            max_houses=self.params.max_houses,
        )

    def get_task_specs(
        self, dataset_dirs: List[str], subset, process_ind, total_processes
    ) -> Hdf5TaskSpecs:
        max_task_specs = self.params.max_task_specs
        if max_task_specs is not None:
            max_task_specs = max_task_specs // total_processes

        tasks = None
        for idx, dataset_dir in enumerate(dataset_dirs):
            if idx == 0:
                tasks = Hdf5TaskSpecs.from_dataset_dir(
                    dataset_dir=dataset_dir,
                    subset=subset,
                    proc_id=process_ind,
                    total_procs=total_processes,
                    max_house_id=self.params.max_houses,
                    max_task_specs=max_task_specs,
                )
            else:
                tasks += Hdf5TaskSpecs.from_dataset_dir(
                    dataset_dir=dataset_dir,
                    subset=subset,
                    proc_id=process_ind,
                    total_procs=total_processes,
                    max_house_id=self.params.max_houses,
                    max_task_specs=max_task_specs,
                )
        return tasks

    @abc.abstractmethod
    def sensors(self) -> List[Sensor]:
        raise NotImplementedError

    @lru_cache(maxsize=None)
    def get_devices(self, mode="train"):
        if torch.cuda.is_available():
            if mode == "train":
                return tuple(range(torch.cuda.device_count()))
            elif mode == "valid":
                return (torch.cuda.device_count() - 1,)
            elif mode == "test":
                return tuple(range(torch.cuda.device_count()))
            else:
                raise NotImplementedError("mode must be train, valid or test")
        else:
            return (torch.device("cpu"),)

    @lru_cache(maxsize=None)
    def get_nprocesses(self, mode="train"):
        num_devices = len(self.get_devices(mode))
        if mode == "train":
            if self.params.num_train_processes == 0:
                return [0] * num_devices
            return evenly_distribute_count_into_bins(self.params.num_train_processes, num_devices)
        elif mode == "valid":
            return [self.num_validation_processes]
        elif mode == "test":
            return evenly_distribute_count_into_bins(self.num_test_processes, num_devices)
        else:
            raise NotImplementedError("mode must be train, valid or test")

    @lru_cache(maxsize=None)
    def get_local_worker_ids(self, mode="train", machine_id=0):
        num_devices = len(self.get_devices(mode))
        return list(
            range(
                num_devices * machine_id,
                num_devices * (machine_id + 1),
            )
        )

    def machine_params(self, mode="train", **kwargs):
        nprocesses = self.get_nprocesses(mode)
        devices = self.get_devices(mode)
        machine_id = kwargs.get("machine_id", 0)
        local_worker_ids = self.get_local_worker_ids(mode=mode, machine_id=machine_id)
        print(f"*****Node-{machine_id} Machine Params [mode={mode}]*****")
        print("NUM PROCESSES", nprocesses)
        print("DEVICES", devices)
        print("LOCAL WORKER IDS", local_worker_ids)
        print("******************************************")

        sensors = self.sensors

        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if len(self.preprocessors()) > 0
            and (
                mode == "train"
                or (
                    (isinstance(nprocesses, int) and nprocesses > 0)
                    or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
                )
            )
            else None
        )

        params = MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=devices,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )
        if mode == "train":
            params.nprocesses = params.nprocesses * self.params.distributed_nodes
            params.devices = params.devices * self.params.distributed_nodes
            params.sampler_devices = params.sampler_devices * self.params.distributed_nodes
            params.set_local_worker_ids(local_worker_ids)

        return params

    def make_sampler_fn(self, **kwargs) -> TaskSampler:
        return MultiTaskSampler(**kwargs)

    def get_sampler_args(
        self,
        mode: Literal["train", "val", "test"],
        process_ind: int,
        total_processes: int,
        dataset_dirs: str,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return task_sampler_args_builder(
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            deterministic_cudnn=deterministic_cudnn,
            mode=mode,
            task_specs=self.get_task_specs(
                dataset_dirs=dataset_dirs,
                subset=mode,
                process_ind=process_ind,
                total_processes=total_processes,
            ),
            houses=self.houses[mode],
            on_server=torch.cuda.is_available(),
            sensors=self.sensors,
            action_space=self.params.actions,
            max_steps=self.params.max_steps,
            max_houses=self.params.max_houses,
            prob_randomize_materials=0.8 if mode == "train" else 0,
        )

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self.get_sampler_args(
            "train",
            dataset_dirs=self.params.dataset_dirs,
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            deterministic_cudnn=deterministic_cudnn,
        )

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        use_jsonl = self.params.use_jsonl_to_load_task_specs
        return self.get_sampler_args(
            "val",
            dataset_dirs=self.params.dataset_dirs,
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            deterministic_cudnn=deterministic_cudnn,
        )

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        use_jsonl = self.params.use_jsonl_to_load_task_specs
        if self.params.test_on_validation:
            return self.valid_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            )
        else:
            return self.get_sampler_args(
                "test",
                dataset_dirs=self.params.dataset_dirs,
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                deterministic_cudnn=deterministic_cudnn,
            )


if __name__ == "__main__":
    MAX_HOUSES = 10
    MAX_TASK_SPECS = 1000
    houses = LazyJsonHouses.from_dir(
        OBJAVERSE_HOUSES_DIR,
        subset="val",
        max_houses=MAX_HOUSES,
    )
    task_specs = LazyJsonTaskSpecs.from_dir(
        "/root/data/ObjectNavType_Poliformer",
        subset="val",
        max_task_specs=MAX_TASK_SPECS,
    )

    sampler_args = task_sampler_args_builder(
        mode="val",
        task_specs=task_specs,
        houses=houses,
        process_ind=0,
        total_processes=1,
        on_server=True,
        sensors=[],
        action_names=SPOCV1ActionSpace().long_action_names,
        max_steps=10,
        devices=[0],
        max_houses=MAX_HOUSES,
    )

    import ipdb

    ipdb.set_trace()
