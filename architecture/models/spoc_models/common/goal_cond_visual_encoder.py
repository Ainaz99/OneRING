from dataclasses import dataclass
from typing import List
from enum import Enum

import torch
import torch.nn as nn
from architecture.models.spoc_models.common.utils import (
    TransformerConfig,
    TransformerEncoder,
)


@dataclass
class GoalCondVisualEncoderConfig:
    fusion_xformer: TransformerConfig = TransformerConfig(3, 512, 8, True)

    visual_features: List[str] = None
    frames_features_dim: int = None
    goal_text_features_dim: int = None
    goal_text_uuid: str = None


class GoalCondVisualEncoder(nn.Module):
    def __init__(self, cfg: GoalCondVisualEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Initialize the visual compressor and visual adapter,
        # where the visual compressor is a CNN that compresses the visual tokens spatially,
        # while the visual adapter project the feature to the transformer dimension
        self.visual_compressor = nn.Sequential(
            nn.Conv2d(self.cfg.frames_features_dim, self.cfg.fusion_xformer.dim, 1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.fusion_xformer.dim, self.cfg.fusion_xformer.dim, 1),
            nn.ReLU(),
        )
        self.visual_adapter = nn.Sequential(
            nn.Linear(self.cfg.fusion_xformer.dim, self.cfg.fusion_xformer.dim),
            nn.LayerNorm(self.cfg.fusion_xformer.dim),
            nn.ReLU(),
        )

        # Initialize the text adapter, which projects the text features to the transformer dimension
        self.text_adapter = nn.Sequential(
            nn.Linear(cfg.goal_text_features_dim, self.cfg.fusion_xformer.dim),
            nn.LayerNorm(self.cfg.fusion_xformer.dim),
            nn.ReLU(),
        )

        # Initialize the fusion transformer encoder.
        # It can take arbitrary number of visual features, goal features, and a fusion token.
        self.fusion_xformer = TransformerEncoder(cfg.fusion_xformer)

        # Initialize the fusion token. This token is corresponding to the state output of the current timestep.
        self.fusion_token = nn.Parameter(0.1 * torch.rand(cfg.fusion_xformer.dim))

        # Initialize the visual sensor tokens.
        # The goal of this token is to allow the model to distinguish between different visual sensors.
        self.visual_features = sorted(self.cfg.visual_features)
        for feature in self.visual_features:
            setattr(
                self,
                f"visual_feature_token_{feature}",
                nn.Parameter(0.1 * torch.rand(cfg.fusion_xformer.dim)),
            )
            

    def adapt_text_features(self, text_features: torch.Tensor):
        return self.text_adapter(text_features)

    def adapt_frames_features(self, frames_features: torch.Tensor):
        B, T, C, H, W = frames_features.shape
        feats = self.visual_compressor(frames_features.reshape(B * T, C, H, W))  # BTxC_xH_xW_
        _, C_, H_, W_ = feats.shape
        feats = feats.reshape(B * T, C_, H_ * W_).permute(0, 2, 1)  # BTxH_W_xC_
        return self.visual_adapter(feats)

    def prepare_features_for_transformer_encoder(
        self,
        frames_features_dict: dict[str, torch.Tensor],
        goals_features_dict: dict[str, torch.Tensor],
    ):
        adapted_text_features = self.adapt_text_features(
            goals_features_dict[self.cfg.goal_text_uuid]
        )
        if len(adapted_text_features.shape) == 3:
            # IL phase, there is a single text feature for the entire episode
            B, L, D = adapted_text_features.shape
            T = frames_features_dict[self.visual_features[0]].shape[1]
            adapted_text_features = (
                adapted_text_features.unsqueeze(1).tile(1, T, 1, 1).reshape(B * T, L, D)
            )
        else:
            # RL phase, it needs to produce text feature at every step, since
            # episodes and episodes are densely concatenated in a batch
            B, T, L, D = adapted_text_features.shape
            adapted_text_features = adapted_text_features.reshape(B * T, L, D)

        concatenated_feats = [
            self.fusion_token.reshape(1, 1, D).tile(B * T, 1, 1),  # BT, 1, D
            adapted_text_features,  # List of BT, L, D
        ]
        for k in self.visual_features:
            frames_features = self.adapt_frames_features(frames_features_dict[k])  # BT, HW, D
            corresponding_camera_token = getattr(self, f"visual_feature_token_{k}")
            concatenated_feats.append(frames_features + corresponding_camera_token)
            
        

        return (
            torch.cat(concatenated_feats, dim=1),
            adapted_text_features.reshape(B, T, L, D)[:, 0],
        )  # BT, L + HW*len(self.visual_features) + 1, D, BxLxD

    def forward(
        self,
        frames_features_dict: dict[str, torch.Tensor],
        goals_features_dict: dict[str, torch.Tensor],
    ):
        B, T, _, H, W = frames_features_dict[self.visual_features[0]].shape
        input_features, adapted_text_features = self.prepare_features_for_transformer_encoder(
            frames_features_dict, goals_features_dict
        )  # BT, L + HW*len(self.visual_features) + 1, D
        fused_feats = self.fusion_xformer(
            input_features
        )  # BT, L + HW*len(self.visual_features) + 1, D
        fused_feats = fused_feats[:, 0, :]  # BTxD
        D = fused_feats.shape[-1]

        return fused_feats.reshape(B, T, D), adapted_text_features  # B, T, D



class RegisteredGoalCondVisualEncoder(Enum):
    GoalCondVisualEncoder = (GoalCondVisualEncoder, GoalCondVisualEncoderConfig())

