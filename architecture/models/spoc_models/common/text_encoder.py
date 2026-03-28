from typing import Literal, Tuple
from dataclasses import dataclass
from open_clip import create_model_from_pretrained
from transformers import T5EncoderModel
import torch
import torch.nn as nn
from enum import Enum


def create_text_encoder(encoder_name: Literal["siglip", "t5-small"]):
    if "siglip" in encoder_name.lower():
        encoder = create_model_from_pretrained(f"hf-hub:timm/{encoder_name}")[0].text
        encoder.output_tokens = True
        return encoder
    elif "t5" in encoder_name.lower():
        return T5EncoderModel.from_pretrained(encoder_name)
    else:
        raise NotImplementedError("Only SigLIP and T5 text encoders are supported.")


@dataclass
class Config:
    model_name: str = "t5-small"

    @property
    def goal_text_features_dim(self):
        if self.model_name == "t5-small":
            return 512
        elif "siglip" in self.model_name.lower():
            return 768
        else:
            raise NotImplementedError("Only SigLIP and T5 text encoders are supported.")


class TextEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if cfg.model_name == "t5-small":
            self.encoder = T5EncoderModel.from_pretrained(cfg.model_name)
        elif "siglip" in cfg.model_name.lower():
            self.encoder = create_model_from_pretrained(f"hf-hub:timm/{cfg.model_name}")[0].text
            self.encoder.output_tokens = True
        else:
            raise NotImplementedError("Only SigLIP and T5 text encoders are supported.")

    def forward(self, goal):
        # goal is the output from T5 tokenizer, including input_ids, attention_mask, etc.
        if "siglip" in self.cfg.model_name.lower():
            with torch.no_grad():
                cls_feats, text_feats = self.encoder(goal)
            return torch.cat([text_feats, cls_feats.unsqueeze(1)], dim=1)
        elif self.cfg.model_name == "t5-small":
            with torch.no_grad():
                feats = self.encoder(**goal).last_hidden_state
            return feats
        else:
            raise NotImplementedError("Only SigLIP and T5 text encoders are supported.")


class TEXT_ENCODER(Enum):
    T5Small = (TextEncoder, Config())
    SigLIPBase = (TextEncoder, Config(model_name="ViT-B-16-SigLIP-256"))
