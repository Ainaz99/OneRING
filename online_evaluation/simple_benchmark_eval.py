"""
Simple script to evaluate a model on a benchmark.
This replaces the complex multi-file evaluation system with a single script.
"""
import argparse
import os
import random
import time
from typing import Dict, List, Any
import numpy as np
import torch
import wandb
import prior

from architecture.models.spoc_models import REGISTERED_MODELS
from online_evaluation.online_evaluation_types_and_utils import (
    NormalizedEvalSample,
    EvalSample,
    eval_sample_to_normalized_eval_sample,
    normalized_eval_sample_to_task_spec,
)
from tasks.multi_task_eval_sampler import MultiTaskSampler
from tasks.task_specs import TaskSpecDatasetList
from data_generation.task_datagen_utils import (
    get_core_task_args,
    add_extra_sensors_to_task_args,
    get_core_sensors,
)
from environment.stretch_controller import StretchController
from environment.action_spaces import SPOCV1ActionSpace, DONE_ACTION, SUB_DONE_ACTION
from environment.evaluation_sensors import TargetbjectWasPickedUp
from environment.agent_parameter_utils import AgentParamRandomizer
from environment.object_nav_sensors import (
    CurrentAgentRoom,
    FinalDistanceToGoal,
    IsObjectVisible,
    NumPixelsVisible,
    OriginalDistanceToGoal,
    TaskNaturalLanguageSpecSensor,
)
from utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
)
import ai2thor.platform
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Simple benchmark evaluation")
    parser.add_argument("--ckpt_path", type=str, default=None, 
                       help="Path to local checkpoint file (or use --hf_repo_id + --hf_checkpoint_file)")
    parser.add_argument("--hf_repo_id", type=str, default=None,
                       help="Hugging Face repository ID (e.g., 'AinazEftekhar/OneRING')")
    parser.add_argument("--hf_checkpoint_file", type=str, default=None,
                       help="Checkpoint filename in Hugging Face repository")
    parser.add_argument("--benchmark_revision", type=str, default="val_spoc_v2_20240802",
                       help="Benchmark revision")
    parser.add_argument("--eval_set_size", type=int, default=None, 
                       help="Number of samples to evaluate (None = all)")
    parser.add_argument("--max_eps_len", type=int, default=-1,
                       help="Max episode length (-1 = use task default)")
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--load_from_pl", action="store_true",
                       help="Load from PyTorch Lightning checkpoint")
    parser.add_argument("--shuffle", action="store_true", default=True,
                       help="Shuffle evaluation samples")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--wandb_logging", action="store_true", 
                       help="Enable wandb logging for evaluation results")
    parser.add_argument("--wandb_project", type=str, default="fpin",
                       help="Wandb project name for logging")
    parser.add_argument("--wandb_entity", type=str, default="prior-ai2",
                       help="Wandb entity")
    parser.add_argument("--output_basedir", type=str, default="./eval_results",
                       help="Base directory for outputs")
    parser.add_argument("--extra_tag", type=str, default="",
                       help="Extra tag for naming output directory")
    return parser.parse_args()


def load_houses(house_set: str):
    """Load houses from the specified dataset."""
    
    if house_set == "objaverse":
        from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
        max_houses_per_split = {"train": 0, "val": int(1e9), "test": 0}
        return prior.load_dataset(
            "spoc-data",
            revision="local",
            path_to_splits=None,
            split_to_path={
                k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
                for k in ["train", "val", "test"]
            },
            max_houses_per_split=max_houses_per_split,
        )["val"]
    else:
        raise ValueError(f"Unknown house_set: {house_set}")


def load_eval_samples(benchmark_revision: str, eval_set_size: int = None, 
                      shuffle: bool = True, seed: int = 123) -> List[NormalizedEvalSample]:
    """Load evaluation samples from benchmark."""

    task_type = "ObjectNavType"
    
    EVAL_TASKS = prior.load_dataset(
        dataset="vida-benchmark",
        revision=benchmark_revision,
        task_types=[task_type],
    )
    
    samples: List[EvalSample] = EVAL_TASKS["val"]
    
    
    sample_ids = list(range(len(samples)))
    if shuffle:
        random.seed(seed)
        random.shuffle(sample_ids)
    
    if eval_set_size is not None:
        sample_ids = sample_ids[:eval_set_size]
    
    normalized_samples = [
        eval_sample_to_normalized_eval_sample(task_type=task_type, sample=samples[i], index=i)
        for i in range(len(samples))
    ]
    
    return [normalized_samples[i] for i in sample_ids]


def evaluate_sample(agent, task, input_sensors: List[str], max_steps: int):
    """Evaluate a single sample."""
    goal = task.task_info["natural_language_spec"]
    print(f"Goal: {goal}")
    agent.reset()
    
    all_actions = []
    success = False
    
    for step in range(max_steps):
        observations = task.get_observations()
        # Filter to only input sensors
        observations = {k: v for k, v in observations.items() if k in input_sensors}
        
        action, probs = agent.get_action(observations, goal)
                
        task.step(task.agent_action_space.get_action_from_meaningful_action_string(action))
        all_actions.append(action)
        
        if task.is_done():
            success = task.is_successful()
            break
    
    # Calculate metrics
    metrics = {
        "success": float(success),
        "eps_len": len(all_actions),
        "real_episode_len": len(task.get_observation_history()) - 1,
    }
    
    return metrics, all_actions


def download_checkpoint_from_hf(repo_id: str, filename: str) -> str:
    """Download checkpoint from Hugging Face and return local path."""
    from huggingface_hub import hf_hub_download
    
    print(f"Downloading checkpoint from Hugging Face: {repo_id}/{filename}")
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=None,  # Use default cache
    )
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} after download")
    
    print(f"Checkpoint downloaded to: {ckpt_path}")
    return ckpt_path


def main():
    args = parse_args()
    
    # Determine checkpoint path
    if args.ckpt_path is None:
        if args.hf_repo_id is None or args.hf_checkpoint_file is None:
            raise ValueError("Must provide either --ckpt_path or both --hf_repo_id and --hf_checkpoint_file")
        ckpt_path = download_checkpoint_from_hf(args.hf_repo_id, args.hf_checkpoint_file)
    else:
        ckpt_path = args.ckpt_path
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup output directory
    os.makedirs(args.output_basedir, exist_ok=True)
    
    # Setup wandb if requested
    wandb_run = None
    if args.wandb_logging:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"eval_ObjectNavType_{int(time.time())}",
        )
        wandb_run = wandb
    
    # Load model
    model_pkg = REGISTERED_MODELS["TuneSpocLlamaModelWTextGoal"]
    model_pkg.config.batch_size = 1
    model_pkg.config.max_seq_len = 600 if args.max_eps_len == -1 else args.max_eps_len
    
    device = f"cuda:{args.gpu_device}" if args.gpu_device >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    agent = model_pkg.model_cls.build_agent(
        cfg=model_pkg.config,
        ckpt_pth=ckpt_path,
        device=device,
        load_from_pl=args.load_from_pl,
    )
    agent.eval()
    
    # Load houses
    print(f"Loading houses from objaverse")
    houses = load_houses("objaverse")
    print(f"Loaded {len(houses)} houses")
    
    # Load evaluation samples
    print(f"Loading evaluation samples: ")
    eval_samples = load_eval_samples(
        args.benchmark_revision,
        args.eval_set_size,
        args.shuffle,
        args.seed,
    )
    print(f"Loaded {len(eval_samples)} evaluation samples")
    
    # Setup task sampler
    max_steps = args.max_eps_len if args.max_eps_len > 0 else 600
    
    extra_sensors = [
        CurrentAgentRoom(),
        NumPixelsVisible(which_camera="manip"),
        NumPixelsVisible(which_camera="nav"),
        TargetbjectWasPickedUp(),
        TaskNaturalLanguageSpecSensor(uuid="goal"),
        IsObjectVisible(which_camera="nav"),
        FinalDistanceToGoal(),
        OriginalDistanceToGoal(),
    ]
    
    task_args = get_core_task_args(max_steps=max_steps, action_space=SPOCV1ActionSpace())
    add_extra_sensors_to_task_args(task_args, extra_sensors)
    
    # Create task specs from eval samples
    task_specs = [normalized_eval_sample_to_task_spec(sample) for sample in eval_samples]
    
    task_sampler = MultiTaskSampler(
        mode="val",
        task_args=task_args,
        houses=houses,
        house_inds=list(range(len(houses))),
        controller_args={
            **STRETCH_ENV_ARGS,
            "platform": (
                ai2thor.platform.OSXIntel64
                if sys.platform.lower() == "darwin"
                else ai2thor.platform.CloudRendering
            ),
        },
        controller_type=StretchController,
        task_spec_sampler=TaskSpecDatasetList(task_specs),
        visualize=False,
        prob_randomize_materials=0,
        device=device if device != "cpu" else None,
    )
    
    # Run evaluation
    print("Starting evaluation...")
    all_metrics = []
    all_results = []
    
    for i, sample in enumerate(eval_samples):
        print(f"Evaluating sample {i+1}/{len(eval_samples)}")
        try:
            task = task_sampler.next_task()
            metrics, actions = evaluate_sample(agent, task, model_pkg.input_sensors, max_steps)
            metrics["sample_id"] = sample["sample_id"]
            all_metrics.append(metrics)
            all_results.append({
                "sample_id": sample["sample_id"],
                "goal": task.task_info["natural_language_spec"],
                "success": metrics["success"],
                "eps_len": metrics["eps_len"],
                "actions": actions,
            })
            print(f"  Success: {metrics['success']}, Steps: {metrics['eps_len']}")
        except Exception as e:
            print(f"  Error evaluating sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Aggregate results
    if all_metrics:
        success_rate = sum(m["success"] for m in all_metrics) / len(all_metrics)
        avg_eps_len = sum(m["eps_len"] for m in all_metrics) / len(all_metrics)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {len(all_metrics)}")
        print(f"Success rate: {success_rate:.3f}")
        print(f"Average episode length: {avg_eps_len:.2f}")
        print("="*50)
        
        # Log to wandb
        if wandb_run:
            wandb_run.log({
                "success_rate": success_rate,
                "avg_eps_len": avg_eps_len,
                "total_samples": len(all_metrics),
            })
            
            # Create results table
            import pandas as pd
            results_df = pd.DataFrame(all_results)
            table = wandb.Table(dataframe=results_df)
            wandb_run.log({"results": table})
    
    # Save results to file
    import json
    results_file = os.path.join(args.output_basedir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
