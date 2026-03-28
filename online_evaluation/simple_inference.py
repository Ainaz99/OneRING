"""
Simple script for inference: takes images and outputs actions.
This is a minimal script that just runs the model forward pass.
"""
import argparse
import os
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple, Any

from architecture.models.spoc_models import REGISTERED_MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="Simple inference: image to action")
    parser.add_argument("--ckpt_path", type=str, default=None,
                       help="Path to local checkpoint file (or use --hf_repo_id + --hf_checkpoint_file)")
    parser.add_argument("--hf_repo_id", type=str, default=None,
                       help="Hugging Face repository ID (e.g., 'AinazEftekhar/OneRING')")
    parser.add_argument("--hf_checkpoint_file", type=str, default=None,
                       help="Checkpoint filename in Hugging Face repository")
    parser.add_argument("--nav_image", type=str, required=True,
                       help="Path to navigation camera image")
    parser.add_argument("--manip_image", type=str, required=True,
                       help="Path to manipulation camera image")
    parser.add_argument("--goal", type=str, required=True,
                       help="Natural language goal/instruction")
    parser.add_argument("--gpu_device", type=int, default=0,
                       help="GPU device ID (-1 for CPU)")
    parser.add_argument("--load_from_pl", action="store_true",
                       help="Load from PyTorch Lightning checkpoint")
    parser.add_argument("--show_probs", action="store_true",
                       help="Show action probabilities")
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    """Load and return image as numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def prepare_observations(nav_image: np.ndarray, manip_image: np.ndarray, 
                        goal: str, last_actions: List[str]) -> Dict[str, Any]:
    """Prepare observations dictionary for the model."""
    return {
        "raw_navigation_camera": nav_image,
        "raw_manipulation_camera": manip_image,
        "goal": goal,
        "last_actions": last_actions,
    }


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
    
    # Setup device
    if args.gpu_device >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.gpu_device}"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load model    
    model_pkg = REGISTERED_MODELS["TuneSpocLlamaModelWTextGoal"]
    model_pkg.config.batch_size = 1
    model_pkg.config.max_seq_len = 600
    
    agent = model_pkg.model_cls.build_agent(
        cfg=model_pkg.config,
        ckpt_pth=ckpt_path,
        device=device,
        load_from_pl=args.load_from_pl,
    )
    agent.eval()
    print("Model loaded successfully")
    
    # Load images
    print(f"Loading navigation image: {args.nav_image}")
    nav_img = load_image(args.nav_image)
    print(f"Loading manipulation image: {args.manip_image}")
    manip_img = load_image(args.manip_image)
    
    # Reset agent state
    agent.reset()
    
    # Prepare observations
    observations = prepare_observations(
        nav_img, 
        manip_img, 
        args.goal,
        last_actions=[""]  # Empty for first step
    )
    observations["traj_index"] = np.array([0], dtype=np.int64)
    
    
    print(observations)
    
    # Get action
    print(f"\nGoal: {args.goal}")
    print("Running inference...")
    
    with torch.no_grad():
        action, probs = agent.get_action(observations, args.goal)
    
    print(f"\n{'='*50}")
    print(f"PREDICTED ACTION: {action}")
    print(f"{'='*50}")
    
    if args.show_probs:
        action_list = agent.get_action_list()
        print("\nAction Probabilities:")
        print("-" * 50)
        # Sort by probability
        action_probs = list(zip(action_list, probs.cpu().numpy()))
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (act, prob) in enumerate(action_probs[:10]):  # Show top 10
            marker = " <-- SELECTED" if act == action else ""
            print(f"{i+1:2d}. {act:30s} {prob:.4f}{marker}")
    
    return action, probs


if __name__ == "__main__":
    main()
