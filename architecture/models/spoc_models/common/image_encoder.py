from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, Sequence
import numpy as np
# from olmo.nn.image_vit import VisionTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained
from enum import Enum

import yaml
import einops

# from olmo.model import VisionTransformer # RMH NOTE: removed to avoid heavy dependencies TODO FIX
from omegaconf import OmegaConf


@dataclass
class Dinov2Config:
    model: str = "dinov2_vits14"
    output_size: Tuple[int, int, int] = (384, 7, 12)
    mean: Optional[Union[np.ndarray, Sequence[float]]] = (0.48145466, 0.4578275, 0.40821073)
    stdev: Optional[Union[np.ndarray, Sequence[float]]] = (0.26862954, 0.26130258, 0.27577711)

    @property
    def frames_features_dim(self):
        return self.output_size[0]


class Dinov2(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        self.model = torch.hub.load("facebookresearch/dinov2", cfg.model)
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (224, 384), f"Expected shape is 224x384; got {x.shape}"
        with torch.no_grad():
            x = self.model.forward_features(x[:, :, :, 3:-3])["x_norm_patchtokens"]
            B, _, D = x.shape  # Bx432x384
            x = x.permute(0, 2, 1)  # Bx384x432
            x = x.reshape(B, D, 16, 27)
            x = self.pool(x)
            return x


@dataclass
class SigLIPConfig(Dinov2Config):
    model: str = "ViT-B-16-SigLIP-256"
    input_size: Tuple[int, int] = (256, 256)
    output_size: Tuple[int, int, int] = (768, 7, 12)
    mean: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)
    stdev: Optional[Union[np.ndarray, Sequence[float]]] = (0.5, 0.5, 0.5)


class SigLIP(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        siglip_full_model = create_model_from_pretrained("hf-hub:timm/{}".format(cfg.model))
        self.model = siglip_full_model[0].visual.trunk
        self.context_length = siglip_full_model[0].context_length
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (256, 256), f"Expected shape is 256x256; got {x.shape}"
        with torch.no_grad():
            x = self.model.forward_features(x)
            B, _, D = x.shape  # Bx256x768
            x = x.permute(0, 2, 1)  # Bx768x256
            x = x.reshape(B, D, 16, 16)
            x = self.pool(x)
            return x


class SigLIPNoPool(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        siglip_full_model = create_model_from_pretrained("hf-hub:timm/{}".format(cfg.model))
        self.model = siglip_full_model[0].visual.trunk
        self.context_length = siglip_full_model[0].context_length
        # self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def forward(self, x):
        assert x.shape[-2:] == (256, 256), f"Expected shape is 256x256; got {x.shape}"
        with torch.no_grad():
            x = self.model.forward_features(x)
            B, _, D = x.shape  # Bx256x768
            x = x.permute(0, 2, 1)  # Bx768x256
            x = x.reshape(B, D, 16, 16)
            # x = self.pool(x)
            return x


@dataclass
class MMOLMODenseCaptioningViTConfig(Dinov2Config):
    model: str = "dense_captioning_vit"
    output_size: Tuple[int, int, int] = (2048, 24, 24)
    mean: Optional[Union[np.ndarray, Sequence[float]]] = (0.48145466, 0.4578275, 0.40821073)
    stdev: Optional[Union[np.ndarray, Sequence[float]]] = (0.26862954, 0.26130258, 0.27577711)

    @property
    def model_dir(self) -> str:
        if torch.cuda.is_available():
            return "/data/output/khzeng"
        else:
            return "/Users/khzeng/Desktop/ckpts"


class MMOLMODenseCaptioningViT(nn.Module):
    def __init__(self, cfg: MMOLMODenseCaptioningViTConfig):
        super().__init__()
        self.cfg = cfg
        with open("{}/{}_config.yaml".format(self.cfg.model_dir, self.cfg.model), "r") as file:
            mmolmo_cfg = yaml.safe_load(file)
        self.mmolmo_cfg = OmegaConf.create(mmolmo_cfg).model
        self.ckpt = "{}/{}.pt".format(self.cfg.model_dir, self.cfg.model)

        # self.model = None  # RMH NOTE: hack to avoid heavy dependencies TODO FIX
        self.model = VisionTransformer(self.mmolmo_cfg)
        self.model.load_state_dict(torch.load(self.ckpt, map_location="cpu"))
        self.image_patch_size = 14

        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def patchify(self, x):
        x = x.permute(0, 2, 3, 1)
        h, w = x.shape[-3:-1]
        return einops.rearrange(
            x,
            "b (dy h dh) (dx w dw) c -> (b dy dx) (h w) (dh dw c)",
            dh=self.image_patch_size,
            dw=self.image_patch_size,
            dy=1,
            dx=1,
            h=h // self.image_patch_size,
            w=w // self.image_patch_size,
        )

    def forward(self, x):
        assert x.shape[-2:] == (336, 336), f"Expected shape is 336x336; got {x.shape}"
        with torch.no_grad():
            x = self.patchify(x)
            x = self.model(x, (336 // self.image_patch_size, 336 // self.image_patch_size))
            x = torch.cat([x[-9], x[-2]], dim=-1)[:, 1:]
            B, _, D = x.shape  # Bx576x2048
            x = x.permute(0, 2, 1)  # Bx2048x576
            x = x.reshape(B, D, 24, 24)
            x = self.pool(x)
            return x


class IMAGE_ENCODER(Enum):
    Dinov2Small = (Dinov2, Dinov2Config())
    Dinov2Base = (Dinov2, Dinov2Config(model="dinov2_vitb14", output_size=(768, 7, 12)))
    SigLIPBase = (SigLIP, SigLIPConfig())
    SigLIPNoPool = (SigLIPNoPool, SigLIPConfig())
    SigLIPLarge = (SigLIP, SigLIPConfig(model="ViT-L-16-SigLIP-256", output_size=(1024, 7, 12)))
    MMOLMODenseCaptioning = (MMOLMODenseCaptioningViT, MMOLMODenseCaptioningViTConfig())
    MMOLMODenseCaptioningPool = (
        MMOLMODenseCaptioningViT,
        MMOLMODenseCaptioningViTConfig(output_size=(2048, 7, 12)),
    )


if __name__ == "__main__":
    image_encoder = IMAGE_ENCODER.Dinov2Small.value
    print(image_encoder)
