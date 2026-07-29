# RING: Robotic Indoor Navigation Generalist

[![Project Website](https://img.shields.io/badge/Website-one--ring--policy.allen.ai-blue)](https://one-ring-policy.allen.ai/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2412.14401)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/AinazEftekhar/OneRING)
[![Data](https://img.shields.io/badge/HuggingFace-Data-yellow?logo=huggingface)](https://huggingface.co/datasets/allenai/ring-data)

**RING (Robotic Indoor Navigation Generalist)** is a generalist policy for indoor visual navigationy trained *solely in simulation* with diverse randomly initialized embodiments at scale (**1 Million** embodiments). RING achieves robust performance on unseen robot platforms (RB-Y1, Stretch RE-1, LoCoBot, Unitree's Go1) in the real world, despite being trained exclusively in simulation **without** any direct exposure to real robot embodiments.

For more details, qualitative examples, and quantitative results, visit our [project website](https://one-ring-policy.allen.ai/).


## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation (eval)](#evaluation-eval)
- [Training (imitation learning)](#training-imitation-learning)
- [Data Generation (datagen)](#data-generation-datagen)
- [Citation](#citation)


## Installation

### Prerequisites

- Python 3.8+ (3.10 recommended)
- CUDA 11.0+ (for GPU support)
- Linux or macOS

### Quick Install

```bash
git clone <repository-url>
cd OneRING

conda create -n ring python=3.10
conda activate ring

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# For CPU-only (not recommended for training, uncomment if needed)
# pip install torch torchvision torchaudio

pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+7c61ad0c3b4a27f169ee02dab01bb83b6303696f

pip install -r requirements.txt
# OR install using pyproject.toml
# pip install -e .

# Verify installation
python -c "import torch, ai2thor, prior, wandb; print('✓ Installation successful')"
```



## Quick Start

### Download Pre-trained Checkpoints

Pre-trained RING checkpoint is available on [Hugging Face](https://huggingface.co/AinazEftekhar/OneRING):

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="AinazEftekhar/OneRING",
    filename="ring_model_step_40356421.ckpt"
)
```

## Evaluation (eval)

The evaluation module provides tools for evaluating the pretrained policy. It supports both full benchmark evaluation and single-step inference.


### 1. Simple Inference (`simple_inference.py`)

Performs single-step inference given 2 camera images and a goal.


```bash

python -m online_evaluation.simple_inference \
    --hf_repo_id AinazEftekhar/OneRING \
    --hf_checkpoint_file ring_model_step_40356421.ckpt \
    --nav_image <path_to_nav_image> \
    --manip_image <path_to_manip_image> \
    --goal <task_instruction> \
    --gpu_device 0 \
    --show_probs
```

``path_to_nav_image`` and ``path_to_manip_image`` specify the path to load the 2 camera observations. If using a single camera you can use the same image. ``task_instruction`` specifies the task in natural language e.g. "go to the TV", "find the toilet" etc. `show_probs` shows action probabilities for all actions.

#### Output

The inference script prints:
- Predicted action
- Action probabilities (if `--show_probs` is enabled)



### 2. Benchmark Evaluation (`simple_benchmark_eval.py`)

Evaluates a model on a full benchmark dataset.


```bash
python -m online_evaluation.simple_benchmark_eval \
    --hf_repo_id AinazEftekhar/OneRING \
    --hf_checkpoint_file ring_model_step_40356421.ckpt \
    --benchmark_revision oct22-15target-allfixesoct31 \
    --output_basedir <output_path> \
     --shuffle \
    --gpu_device 0
```

## Training (imitation learning)
Training uses PyTorch Lightning and logs to Weights & Biases by default.


### Basic usage

```bash
export PYTHONPATH="./"
export OBJAVERSE_DATA_DIR=<path_to_objaverse_data> \
python -m training.offline.train_spoc_pl \
    --max_samples 1000000 \
    --eval_max_samples 100 \
    --eval_every 300 \
    --sliding_window 100 \
    --per_gpu_batch 16 \
    --lr 0.0002 \
    --data_dir /path/to/datasets \
    --dataset_version ObjectNavType \
    --model SpocLlamaModelWTextGoal \
    --precision 16-mixed \
    --loss action \
    --max_epochs 400 \
    --output_dir /path/to/results \
    --num_nodes 1 \
    --extra_tag SpocLlamaModelWTextGoal
```

### What the key arguments mean
- `--model`: Registered model to train. 
- `--data_dir`: Root directory containing the offline dataset files.
- `--output_dir`: Directory for logs and checkpoints. A subfolder is created per run.
- `--per_gpu_batch`: Batch size per GPU. Effective batch = per_gpu_batch × GPUs × nodes.
- `--max_epochs`: Number of training epochs.
- `--eval_every`: Validation frequency (in training steps).
- `--save_every`: Checkpoint save frequency (in training steps).
- `--precision`: Use `32-true` (default) or `16-mixed` for mixed precision on GPU.
- `--sliding_window`: Temporal context length used by the dataset loader.







## Data Generation (datagen)

The data generation module creates offline expert demonstration datasets for training RING. It uses multiprocessing to efficiently generate trajectories across multiple houses and tasks with randomized embodiments.

 Ensure Objaverse or Procthor house datasets are available. Place them in the directory specified by `OBJAVERSE_DATA_DIR`. See the project documentation for dataset download instructions

### Overview

The datagen system:
- Generates expert trajectories using path planners
- Uses multiprocessing with queues for parallel generation
- Saves trajectories as HDF5 files with associated metadata
- Enables embodiment randomization for diverse training data

### Usage


```bash
export OBJAVERSE_DATA_DIR=<path_to_objaverse_data> \
export PYTHONPATH="./" \
python -m data_generation.multi_task_stretch_mpqueue \
    --save_dir <output_directory> \
    --dataset_name "ObjectNavType" \
    --split <train|val|test> \
    --house_repeats <number_of_trajectories_per_house> \
    --max_steps <max_steps_per_trajectory> \
    --house_dataset objaverse \
    --task_type "ObjectNavType" \
    --action_space_type "spoc" \
    --action_space_name "spocV1"
```

### Arguments

- `--save_dir`: Directory to save the generated dataset
- `--split`: Dataset split - `train`, `val`, or `test`
- `--house_repeats`: Number of trajectories to generate per house
- `--max_steps`: Maximum number of steps per trajectory (default: 1000)
- `--stochastic_controller`: Whether to use stochastic controller variant (default: True)
- `--closed_type`: Limit target object types for backwards compatibility (default: False)



### Output Format

The datagen system saves:
- **HDF5 files**: Trajectory data including observations, actions, and metadata
- **MP4 files**: Video recordings of trajectories (if enabled)
- **Text files**: Metadata and configuration information
- **Metrics JSON**: Generation statistics (success/failure counts)

```
<save_dir>/
  <dataset_name>/
    <split>/
      <house_id>/
        <trajectory_id>.hdf5
        <trajectory_id>.mp4
        <trajectory_id>.txt
```

### Architecture

- **Manager** (`manager_single_machine_mpqueue.py`): Coordinates workers and manages job queues
- **Worker** (`worker.py`): Processes individual trajectories using path planners
- **Path Planners** (`path_planners.py`): Generate expert trajectories for tasks
- **Task Samplers**: Sample tasks from houses and object configurations

### Data Generation Issues

- **Out of Memory**: Reduce `--house_repeats` or number of workers
- **Planner Failures**: Some houses may be invalid for certain tasks - these are logged and skipped



## Citation

If you use RING in your research, please cite:

```bibtex
@article{ring2024,
  title={RING: Robotic Indoor Navigation Generalist},
  author={Eftekhar, Ainaz and Hendrix, Rose and Weihs, Luca and Duan, Jiafei and Caglar, Ege and Salvador, Jordi and Herrasti, Alvaro and Han, Winson and VanderBilt, Eli and Kembhavi, Aniruddha and Farhadi, Ali and Krishna, Ranjay and Ehsani, Kiana and Zeng, Kuo-Hao},
  journal={arXiv preprint},
  year={2024}
}
```

---

