import argparse
import os
from PIL import Image
import random
import warnings
from typing import Optional, Any, Mapping, Dict

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import F1Score
from torchmetrics.aggregation import SumMetric

from allenact.utils.misc_utils import str2bool
from architecture.models.spoc_models import REGISTERED_MODELS
from training.offline.dataset_mixtures import get_mixture_by_name
from training.offline.ilearn_dataset import iLearnMultitaskDataset
from training.offline.train_utils import get_latest_local_ckpt_pth

import platform

from utils.visualization_utils import create_multiline_text_image

from vida_constants import WANDB_DEFAULT_PROJECT_NAME



def arg_parser_for_offline_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SpocModel")
    parser.add_argument("--loss", type=str, default="action")
    parser.add_argument("--dataset_version", type=str, default="object_nav_v0.3")
    parser.add_argument("--data_dir", type=str, default="/data/datasets")
    parser.add_argument("--output_dir", type=str, default="/data/results")
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--eval_max_samples", type=int, default=1600)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--wandb_logging", default=True, type=str2bool)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_video_every", type=int, default=2000)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--per_gpu_batch", type=int, default=16)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sliding_window", type=int, default=50)
    parser.add_argument("--init_prob_sample_last_steps", type=float, default=0.0)
    parser.add_argument("--final_prob_sample_last_steps", type=float, default=0.0)
    parser.add_argument("--reduce_action_redundancy", type=str2bool, default=False)

    parser.add_argument("--precision", type=str, default="32-true", choices=["32-true", "16-mixed"])
    # resume training from last local checkpoint
    parser.add_argument("--resume_local", action=argparse.BooleanOptionalAction)
    # resume from specified run id and step
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_non_strict_ckpt_loading", action=argparse.BooleanOptionalAction)
    parser.add_argument("--restart_optimizer", action=argparse.BooleanOptionalAction)
    # initialize model from a specified run_id and step
    parser.add_argument("--init_model", action=argparse.BooleanOptionalAction)
    # specify run id for --resume or --init_model
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument(
        "--output_sensors",
        nargs="+",
        default=[],
    )
    parser.add_argument("--extra_tag", type=str, default="")
    parser.add_argument(
        "--use_tokenized_action", type=bool, default=False, help="Whether to use tokenized actions"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="Path to the tokenizer to use"
    )
    return parser


class AdamWSkipLoadStateDict(optim.AdamW):
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        warnings.warn("AdamWSkipLoadStateDict IS IGNORING A REQUEST TO LOAD A STATE DICT")
        return


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.use_non_strict_ckpt_loading = args.use_non_strict_ckpt_loading
        self.restart_optimizer = args.restart_optimizer
        # IL training doesn't use KV Cache, so let's keep it 1x1 for better memory usage
        REGISTERED_MODELS[args.model].config.batch_size = 1
        REGISTERED_MODELS[args.model].config.max_seq_len = 1
        self.model = REGISTERED_MODELS[args.model].build_model()
        self.preproc = self.model.build_preproc(self.model.cfg.preproc_config)
        self.loss_name = args.loss
        self.args = args
        self.metrics = self.get_metrics()
        self.train_steps = 0
        self.num_frames = 0
        self.frames_metric = SumMetric()
        self.log_video_every = args.log_video_every
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def on_fit_start(self):
        self.preproc.to(self.device)
        if platform.system() == "Darwin":
            self.preproc.to(torch.device("cpu"))
            self.model = self.model.to(self.preproc.device)
        self.frames_metric.reset()

    def log_videos(self, batch, outputs, train_or_val):
        items_to_log = random.choices(range(len(batch)), k=min(10, len(batch)))
        columns = ["Task", "input_sensors", "Actions_gt", "Actions_pred", "Sensor_path"]
        data = []
        for item_to_log in items_to_log:
            batch_item = batch[item_to_log]

            output_item = outputs["actions_logits"][item_to_log]
            pred = output_item.argmax(-1).cpu().tolist()
            actions_gt = list(batch_item["input_sensors"]["actions"])
            actions_pred = [
                self.preproc.cfg.action_space.full_action_list[action_idx]
                for action_idx in pred
            ]

            task = batch_item["input_sensors"]["goal"]

            def combine_observations_and_gt_action_and_save_path(cameras, actions_gt, actions_pred):
                # Convert camera tensors to numpy arrays
                # warped_cameras = [
                #     warp_video_gpu(
                #         cam,
                #         randomize_distortion_parameters=True,
                #         output_shape=(MODEL_43_HEIGHT, MODEL_43_WIDTH),
                #     )
                #     for cam in cameras
                # ]
                # cameras = [cam for cam in warped_cameras]
                cameras = [cam.cpu().numpy() for cam in cameras]
                T, w, h, c = cameras[0].shape

                # Create action cam with updated height
                texts_to_write = [
                    f"gt:{actions_gt[i]}\n pred: {actions_pred[i]}" for i in range(len(cameras[0]))
                ]

                # Create a blank image for missing camera slots
                blank_image = np.zeros((T, w, h, c), dtype=cameras[0].dtype)

                # Arrange cameras in a 2x2 grid
                grid = []
                for i in range(0, len(cameras), 2):
                    row = cameras[i : i + 2]
                    if len(row) < 2:
                        row.append(blank_image)
                    grid.append(np.concatenate(row, axis=2))

                # Stack rows to form the final grid
                full_cam = np.concatenate(grid, axis=1)
                # make action cam that is the same size as the camera
                action_cam = [
                    create_multiline_text_image(
                        texts_to_write[i], full_cam[i].shape[1], full_cam[i].shape[0]
                    )
                    for i in range(len(cameras[0]))
                ]
                action_cam = np.stack(action_cam, axis=0)
                full_cam = np.concatenate([full_cam, action_cam], axis=2)
                full_cam = np.transpose(full_cam, (0, 3, 1, 2))
                return wandb.Video(full_cam, fps=5)

            cameras = []
            for cam in [
                "raw_navigation_camera",
                "raw_manipulation_camera",
            ]:
                if cam in batch_item["input_sensors"]:
                    cameras.append(batch_item["input_sensors"][cam])
            video = combine_observations_and_gt_action_and_save_path(
                cameras,
                actions_gt,
                actions_pred,
            )

            assert "raw_navigation_camera" in batch_item
            sensor_path = batch_item["raw_navigation_camera"]
            data.append([task, video, actions_gt, actions_pred, sensor_path])

        if hasattr(self.logger, "log_table"):
            self.logger.log_table(
                key=f"video_action_table/{train_or_val}/{self.train_steps}",
                columns=columns,
                data=data,
            )

    def compute_loss(self, logits, actions):
        B, T, C = logits.shape
        return self.loss(logits.reshape(-1, C), actions.reshape(-1))

    def forward_batch(self, batch):
        proc_batch = self.preproc.process(batch)
        outputs = self.model(observations=proc_batch, prev_actions=proc_batch["last_actions"])
        loss = self.compute_loss(outputs, proc_batch["actions"])
        outputs = {"actions_logits": outputs, "action_loss": loss, "loss": loss}
        return outputs, proc_batch

    def training_step(self, batch, batch_idx):
        self.train_steps += 1
        outputs, proc_batch = self.forward_batch(batch)
        self.frames_metric.update(proc_batch["lengths"])
        train_frames = 0
        if self.train_steps % 10 == 0:
            train_frames = self.frames_metric.compute()

        losses = dict()
        for k, v in outputs.items():
            if "loss" in k:
                losses[f"{k}/train"] = v

        self.log_dict(
            {
                **losses,
                "train_steps": float(self.train_steps),
                "train_frames": train_frames,
                "current_prob_to_sample_last_steps": float(
                    min([b["prob_sample_last_steps"] for b in batch])
                ),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        if self.train_steps % self.log_video_every == 0:
            self.log_videos(batch, outputs, "train")
        return outputs

    def get_metrics(self):
        metrics = dict()
        if self.model.cfg.action_space.get_num_actions() > 0 and self.loss_name == "action":
            metrics["f1score_weighted"] = F1Score(
                task="multiclass",
                num_classes=self.model.cfg.action_space.get_num_actions(),
                ignore_index=-1,
                average="weighted",
            )
            metrics["f1score_macro"] = F1Score(
                task="multiclass",
                num_classes=self.model.cfg.action_space.get_num_actions(),
                ignore_index=-1,
                average="macro",
            )
            metrics["f1score"] = F1Score(
                task="multiclass",
                num_classes=self.model.cfg.action_space.get_num_actions(),
                ignore_index=-1,
                average=None,
            )
        return metrics

    def on_train_epoch_start(self) -> None:
        prob_decay_size = (
            self.args.init_prob_sample_last_steps - self.args.final_prob_sample_last_steps
        ) / args.max_epochs
        current_prob = (
            self.args.init_prob_sample_last_steps - prob_decay_size * self.trainer.current_epoch
        )
        next_prob = self.args.init_prob_sample_last_steps - prob_decay_size * (
            self.trainer.current_epoch + 1
        )
        # 4 is the current number of workers we use in the dataloader
        self.trainer.train_dataloader.dataset.init_prob_sample_last_steps(
            init_prob=current_prob,
            final_prob=next_prob,
            num_workers=4,
            num_gpu_per_node=max(torch.cuda.device_count(), 1),
            num_node=self.args.num_nodes,
        )

    def on_validation_epoch_start(self):
        for metric_name, metric in self.metrics.items():
            self.metrics[metric_name] = metric.to(self.device)
            if platform.system() == "Darwin":
                target_device = torch.device("cpu")
                self.metrics[metric_name] = metric.to(target_device)

    def validation_step(self, batch, batch_idx):
        outputs, proc_batch = self.forward_batch(batch)
        losses = dict()
        for k, v in outputs.items():
            if "loss" in k:
                losses[f"{k}/val"] = v

        self.log_dict(
            {
                **losses,
                "train_steps": float(self.train_steps),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )

        if self.loss_name == "quantized_goal_pose":
            # gt = proc_batch["last_goal_agent_frame"]
            # pred = outputs["goal_pose_logits"].argmax(-1)
            gt = proc_batch["actions"]
            pred = outputs["actions_logits"].argmax(-1)
        elif self.loss_name == "action":
            gt = proc_batch["actions"]
            pred = outputs["actions_logits"].argmax(-1)

        if batch_idx == 0:
            self.log_videos(batch, outputs, "val")

        for metric_name in self.metrics:
            self.metrics[metric_name](pred, gt)

    def on_validation_epoch_end(self):
        metrics_to_log = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == "f1score":
                if self.loss_name == "quantized_goal_pose":
                    continue
                action_f1scores = metric.compute()
                for action_idx, action_name in enumerate(
                    self.preproc.cfg.action_space.full_action_list
                ):
                    metrics_to_log[f"{metric_name}/{action_name}/val"] = action_f1scores[action_idx]
            else:
                metrics_to_log[f"{metric_name}/val"] = metric.compute()

        self.log_dict(
            dict(**metrics_to_log, train_steps=self.train_steps),
            sync_dist=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        for metric in self.metrics.values():
            metric.reset()

    def configure_optimizers(self):
        if self.restart_optimizer:
            return AdamWSkipLoadStateDict(self.model.parameters(), lr=self.args.lr)
        else:
            return optim.AdamW(self.model.parameters(), lr=self.args.lr)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["train_steps"] = self.train_steps
        self.logger._checkpoint_name = f"ckpt-{self.logger.experiment.id}-{self.train_steps}"

    def on_load_checkpoint(self, checkpoint):
        self.train_steps = checkpoint["train_steps"]
        self.trainer.fit_loop.epoch_progress.current.completed = checkpoint["epoch"]

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        state_dict = {
            k.replace(
                "model.visual_encoder.image_encoder.model.visual.trunk",
                "model.visual_encoder.image_encoder.model",
            ): v
            for k, v in state_dict.items()
        }
        state_dict = {
            k.replace(
                "model.visual_encoder.image_encoder.model.text.transformer",
                "model.visual_encoder.text_encoder.transformer",
            ): v
            for k, v in state_dict.items()
        }
        for k in [
            "logit_scale",
            "logit_bias",
            "text.positional_embedding",
            "text.token_embedding.weight",
            "text.ln_final.weight",
            "text.ln_final.bias",
            "text.text_projection.weight",
            "text.text_projection.bias",
        ]:
            k = f"model.visual_encoder.image_encoder.model.{k}"
            if k in state_dict:
                del state_dict[k]

        assert strict is None or strict == (not self.use_non_strict_ckpt_loading)
        strict = not self.use_non_strict_ckpt_loading

        return super().load_state_dict(state_dict, strict=strict)


def identity_collate(batch):
    return [sample for sample in batch if sample is not None]


def get_dataloader(subset: str, args):
    input_sensors = REGISTERED_MODELS[args.model].input_sensors
    action_space = REGISTERED_MODELS[args.model].config.action_space
    dataset = iLearnMultitaskDataset(
        base_data_dir=args.data_dir,
        dataset_names=get_mixture_by_name(args.dataset_version),
        subset=subset,  # temporary
        max_samples=args.max_samples if subset == "train" else args.eval_max_samples,
        proc_idx=0,  # can't use with DDP
        num_procs=1,  # can't use with DDP
        sliding_window=args.sliding_window,
        input_sensors=input_sensors,
        output_sensors=args.output_sensors,
        reduce_action_redundancy=args.reduce_action_redundancy if subset == "train" else False,
        action_space=action_space,
    )

    return DataLoader(
        dataset,
        batch_size=args.per_gpu_batch,
        num_workers=4 if torch.cuda.is_available() else 1,
        prefetch_factor=2,
        collate_fn=identity_collate,
        persistent_workers=False,
        pin_memory=True,
    )


def launch_training(args):
    local_world_size = max(torch.cuda.device_count(), 1)

    # create data loaders
    data_loaders = dict(
        train=get_dataloader("train", args),
        val=get_dataloader("val", args),
    )

    # set args
    args.num_datasets = len(data_loaders["train"].dataset.dataset_names)
    # max_samples is per dataset, so we need to multiply by num_datasets
    args.max_samples = min(
        args.max_samples * args.num_datasets,
        len(data_loaders["train"].dataset),
    )
    args.exp_name = ",".join(
        [
            f"pl-model={args.model}",
            f"dataset={args.dataset_version}",
            f"batch_size={args.per_gpu_batch * local_world_size * args.num_nodes}",
            f"lr={args.lr}",
            f"scale={args.max_samples}",
            f"extra_tag={args.extra_tag}",
        ]
    )
    args.exp_dir = os.path.join(args.output_dir, args.exp_name)

    # create logger
    logger: Optional[pl.loggers.wandb.WandbLogger] = None
    if args.wandb_logging:
        logger = pl.loggers.wandb.WandbLogger(
            project=WANDB_DEFAULT_PROJECT_NAME,
            entity="prior-ai2",
            name=args.exp_name,
            save_dir=args.output_dir,
            config=vars(args),
            log_model="all",
        )

    if args.init_model:
        init_model_dir = os.path.join(args.exp_dir, args.run_id, str(args.step))
        logger.download_artifact(
            f"prior-ai2/{WANDB_DEFAULT_PROJECT_NAME}/ckpt-{args.run_id}-{args.step}:latest",
            save_dir=init_model_dir,
        )
        args.ckpt_pth = os.path.join(init_model_dir, "model.ckpt")
    else:
        args.ckpt_pth = None

    # create model
    lit_model = LitModel(args)

    # create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.exp_dir,
        filename="checkpoint_{train_steps:.0f}",
        save_top_k=-1,
        verbose=True,
        every_n_train_steps=args.save_every,
    )

    # create trainer and train
    if torch.cuda.is_available():
        devices = local_world_size
        accelerator = "gpu"
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        devices = accelerator = strategy = "auto"
        ccelerator = "cpu"
        devices = 1
        accelerator = "auto"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        args.precision = "32-true"  # mixed precision doesn't work on cpu

    # trainer = pl.Trainer(devices=devices,num_nodes=args.num_nodes,accelerator=accelerator,strategy=strategy,callbacks=[checkpoint_callback],default_root_dir=args.output_dir,val_check_interval=args.eval_every,log_every_n_steps=10,max_epochs=args.max_epochs,logger=logger,precision=args.precision,)

    trainer = pl.Trainer(
        devices=devices,
        num_nodes=args.num_nodes,
        accelerator=accelerator,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        default_root_dir=args.output_dir,
        val_check_interval=args.eval_every,
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        logger=logger,
        precision=args.precision,
    )

    resume_ckpt_path = None
    if args.resume:
        ckpt_dir = os.path.join(args.exp_dir, args.run_id, str(args.step))
        resume_ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
        if not os.path.exists(resume_ckpt_path):
            logger.download_artifact(
                f"prior-ai2/{WANDB_DEFAULT_PROJECT_NAME}/ckpt-{args.run_id}-{args.step}:latest",
                save_dir=ckpt_dir,
            )
        print("Resuming from:", resume_ckpt_path)
    elif args.resume_local:
        resume_ckpt_path = get_latest_local_ckpt_pth(args.exp_dir)
        if resume_ckpt_path is None:
            print("No local ckpt found. Training from scratch.")
        else:
            print("Resuming from local ckpt:", resume_ckpt_path)
    else:
        print(
            'Training from scratch. Set "--resume" (along with "--run_id" and "--step") to resume from a checkpoint.'
        )

    trainer.fit(
        lit_model,
        data_loaders["train"],
        data_loaders["val"],
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    args = arg_parser_for_offline_training().parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.hub._validate_not_a_forked_repo = (
        lambda a, b, c: True
    )  # KE: This is for getting around the http limit rate error. From https://github.com/pytorch/vision/issues/4156#issuecomment-886005117
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # reduced matmul precision for NVIDIA A6000 GPUs
    if torch.cuda.is_available():
        if args.precision == "16-mixed":
            torch.set_float32_matmul_precision("medium")
        elif args.precision == "32-true":
            pass
        else:
            raise NotImplementedError(f"Unknown precision {args.precision}")
    launch_training(args)
