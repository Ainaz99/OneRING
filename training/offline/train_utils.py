import os
import glob
import socket
from contextlib import closing

import torch
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score

from training.offline.dataset_mixtures import get_mixture_by_name
from training.offline.ilearn_dataset import iLearnMultitaskDataset


def process_batch_for_model(batch, preprocessor, input_sensors):
    processed_batch = preprocessor.process(batch)
    model_inputs = dict(
        padding_mask=processed_batch["padding_mask"],
        goals=processed_batch["goals"],
        time_ids=processed_batch["time_ids"],
        **{sensor_name: processed_batch[sensor_name] for sensor_name in input_sensors},
    )
    return model_inputs, processed_batch


def save_and_upload_checkpoint(wandb, ckpt_dict, step, args):
    ckpt_pth = os.path.join(args.output_dir, "checkpoint.pth")
    torch.save(ckpt_dict, ckpt_pth)
    wandb.save(ckpt_pth, policy="live")
    step_ckpt_pth = os.path.join(args.output_dir, f"checkpoint_{step}.pth")
    torch.save(ckpt_dict, step_ckpt_pth)
    wandb.save(step_ckpt_pth, policy="live")


def put_model_on_train(model, device):
    model.train()
    if device == "cpu":
        model.visual_encoder.image_encoder.eval()
    else:
        model.module.visual_encoder.image_encoder.eval()
    return model


def get_eval_metric_functions(num_classes, device):
    metric_fn = dict()
    metric_fn["precision"] = Precision(
        task="multiclass", num_classes=num_classes, ignore_index=-1, average=None
    ).to(device)
    metric_fn["recall"] = Recall(
        task="multiclass", num_classes=num_classes, ignore_index=-1, average=None
    ).to(device)
    metric_fn["f1score"] = F1Score(
        task="multiclass", num_classes=num_classes, ignore_index=-1, average=None
    ).to(device)
    metric_fn["f1score_weighted"] = F1Score(
        task="multiclass", num_classes=num_classes, ignore_index=-1, average="weighted"
    ).to(device)
    metric_fn["f1score_macro"] = F1Score(
        task="multiclass", num_classes=num_classes, ignore_index=-1, average="macro"
    ).to(device)
    metric_fn["rooms_seen_f1score"] = F1Score(
        task="multiclass", num_classes=20, ignore_index=19, average="weighted"
    ).to(device)
    metric_fn["room_current_seen_f1score"] = F1Score(
        task="multiclass", num_classes=3, ignore_index=2, average="weighted"
    ).to(device)
    return metric_fn


def calculate_metrics(outputs, batch, metric_fn):
    gt = batch["actions"]
    pred = outputs["logits"].argmax(-1)
    metrics = {name: fn(pred, gt) for name, fn in metric_fn.items() if "room" not in name}

    if "rooms_seen_logits" in outputs and "rooms_seen" in batch:
        rooms_seen_pred = outputs["rooms_seen_logits"].argmax(-1)
        rooms_seen_gt = batch["rooms_seen"][:, 1:]
        metrics["rooms_seen_f1score"] = metric_fn["rooms_seen_f1score"](
            rooms_seen_pred, rooms_seen_gt
        )
    if "room_current_seen_logits" in outputs and "room_current_seen" in batch:
        room_current_seen_pred = outputs["room_current_seen_logits"].argmax(-1)
        room_current_seen_gt = batch["room_current_seen"][:, 1:]
        metrics["room_current_seen_f1score"] = metric_fn["room_current_seen_f1score"](
            room_current_seen_pred, room_current_seen_gt
        )
    return metrics


def get_dataloader(subset, args, rank, num_procs):
    dataset = iLearnMultitaskDataset(
        base_data_dir=args.data_dir,
        dataset_names=get_mixture_by_name(args.dataset_version),
        subset=subset,
        max_samples=args.max_samples if subset == "train" else args.eval_max_samples,
        proc_idx=rank,
        num_procs=num_procs,
        sliding_window=args.sliding_window,
        input_sensors=args.input_sensors,
    )
    assert len(dataset) > 0
    return DataLoader(
        dataset,
        batch_size=args.per_gpu_batch,
        num_workers=8,
        prefetch_factor=4,
        collate_fn=identity_collate,
        persistent_workers=False,
        shuffle=True,
    )


def identity_collate(batch):
    return [sample for sample in batch if sample is not None]


def load_weights(model, optimizer, args, device):
    start_step = 0
    start_epoch = 0
    if args.ckpt_pth is not None:
        print(f"Loading checkpoint: {args.ckpt_pth}")
        ckpt = torch.load(args.ckpt_pth, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
    elif args.visual_encoder_ckpt_pth is not None:
        print(f"Loading visual encoder checkpoint: {args.visual_encoder_ckpt_pth}")
        ckpt = torch.load(args.visual_encoder_ckpt_pth, map_location=device)
        visual_encoder_params = {
            ".".join(k.split(".")[1:]): v for k, v in ckpt["model"].items() if "visual_encoder" in k
        }
        model.visual_encoder.load_state_dict(visual_encoder_params)
    else:
        print("No checkpoint to load")

    if args.ckpt_pth is not None:
        optimizer.load_state_dict(ckpt["optim"])
    return model, optimizer, start_step, start_epoch


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def load_allenact_ckpt(model, ckpt_pth):
    print(f"Loading ckpt {ckpt_pth} ...")
    ckpt_state_dict = torch.load(ckpt_pth, map_location="cpu")["model_state_dict"]
    model.load_state_dict(ckpt_state_dict)


def load_pl_ckpt(model, ckpt_pth, ckpt_prefix="model.", verbose=False):
    print(f"Loading ckpt {ckpt_pth} using ckpt_prefix='{ckpt_prefix}' ...")
    ckpt_state_dict = torch.load(ckpt_pth, map_location="cpu")["state_dict"]

    # init new state dict with curr state dict
    new_state_dict = model.state_dict()

    params_to_load = [k for k in new_state_dict.keys() if ckpt_prefix + k in ckpt_state_dict]
    for k in params_to_load:
        new_state_dict[k] = ckpt_state_dict[ckpt_prefix + k]

    model.load_state_dict(new_state_dict)
    if verbose:
        print("-" * 80)
        print(
            "Parameters found and loaded from the checkpoint:",
            params_to_load,
        )
        print("-" * 80)

    params_in_model_not_in_ckpt = [
        k for k in new_state_dict.keys() if ckpt_prefix + k not in ckpt_state_dict
    ]
    if len(params_in_model_not_in_ckpt) > 0:
        print("-" * 80)
        print(
            "Parameters that are present in model but not present in checkpoint:",
            params_in_model_not_in_ckpt,
        )
        print("-" * 80)

    params_in_ckpt_not_in_model = [
        k[len(ckpt_prefix) :]
        for k in ckpt_state_dict
        if k[len(ckpt_prefix) :] not in new_state_dict
    ]
    if len(params_in_ckpt_not_in_model):
        print("-" * 80)
        print(
            f"Parameters present in checkpoint not present in model (removing ckpt_prefix='{ckpt_prefix}'):",
            params_in_ckpt_not_in_model,
        )
        print("-" * 80)


def get_latest_local_ckpt_pth(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if len(ckpts) == 0:
        return None

    ckpts = sorted(ckpts, key=lambda x: int(x.split(".")[0].split("=")[-1]))
    return os.path.join(ckpt_dir, ckpts[-1])


def get_latest_rl_local_ckpt_pth(ckpt_dir):
    # Use glob to find all .ckpt files recursively
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "**", "*.pt"), recursive=True)

    if not ckpt_files:
        return None

    # Sort the checkpoint files based on the step number extracted from the filename
    ckpt_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))

    # Return the path to the latest checkpoint
    return ckpt_files[-1]
