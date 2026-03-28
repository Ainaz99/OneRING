from typing import Optional, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from environment.action_spaces import SPOCV1ActionSpace
from architecture.models.spoc_models.common.utils import (
    TransformerConfig,
    TransformerDecoder,
    PositionalEncoder,
)
from architecture.models.spoc_models.common.llama import (
    TransformerDecoder as LlamaDecoder,
    ModelArgs as LlamaConfig,
)
from architecture.models.spoc_models.common.goal_cond_llama import (
    TransformerDecoder as GoalCondLlamaDecoder,
    ModelArgs as GoalCondLlamaConfig,
)

def create_causal_mask(T: int, device: torch.device):
    return torch.triu(torch.full([T, T], float("-inf"), device=device), diagonal=1)


@dataclass
class ActionDecoderConfig:
    decoder: TransformerConfig = TransformerConfig(3, 512, 8, True)
    max_length: int = 1000
    an_object_is_in_hand: bool = True
    action_space: SPOCV1ActionSpace = None


class ActionDecoder(nn.Module):
    def __init__(
        self,
        cfg: ActionDecoderConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.decoder = TransformerDecoder(cfg.decoder)

        self.time_encoder = PositionalEncoder(self.cfg.decoder.dim, self.cfg.max_length)

        # if num_actions=20; then 0-19 are actions, 20 is for padding, and 21 is for start token
        self.last_actions_embed = nn.Embedding(
            self.cfg.action_space.get_num_actions() + 2,
            self.cfg.decoder.dim,
            padding_idx=self.cfg.action_space.get_num_actions(),  # this is because self.cfg.action_space.get_num_actions() returns 20
        )
        self.last_actions_embed.weight.data.uniform_(-0.01, 0.01)

        if cfg.an_object_is_in_hand:
            self.object_in_hand_embed = nn.Embedding(3, self.cfg.decoder.dim)
            self.object_in_hand_embed.weight.data.uniform_(-0.01, 0.01)

    def get_input_embedding_per_timestep(
        self,
        goal_cond_visual_feats: torch.Tensor,  # B, T, D
        last_actions: torch.Tensor,  # B, T
        time_ids: torch.Tensor,  # B, T
        an_object_is_in_hand: Optional[torch.Tensor] = None,  # B, T
    ) -> torch.Tensor:

        last_actions_enc = self.last_actions_embed(last_actions)
        state_embed = goal_cond_visual_feats + last_actions_enc

        if an_object_is_in_hand is not None:
            object_in_hand_enc = self.object_in_hand_embed(an_object_is_in_hand)
            state_embed = state_embed + object_in_hand_enc

        time_enc = self.time_encoder(time_ids)
        state_embed = state_embed + time_enc

        return state_embed

    def decode(self, state_embed, text_feats, causal_mask=None, padding_mask=None):
        if causal_mask is None:
            causal_mask = create_causal_mask(state_embed.shape[1], state_embed.device)
        decoder_output = self.decoder(
            tgt=state_embed,
            memory=text_feats,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        return decoder_output

    def forward(
        self,
        goal_cond_visual_feats: torch.Tensor,  # B, T, D
        goals_features: torch.Tensor,  # B, L, D
        last_actions: torch.Tensor,  # B, T
        time_ids: torch.Tensor,  # B, T
        causal_mask: Optional[torch.Tensor] = None,  # B, T, T
        padding_mask: Optional[torch.Tensor] = None,  # B, T
        an_object_is_in_hand: Optional[torch.Tensor] = None,  # B, T
        kv_cache_step_counter: Optional[int] = None,  # Only used for Llama decoder
    ):
        state_embed = self.get_input_embedding_per_timestep(
            goal_cond_visual_feats,
            last_actions,
            time_ids,
            an_object_is_in_hand,
        )

        return self.decode(state_embed, goals_features, causal_mask, padding_mask)


@dataclass
class LlamaActionDecoderConfig:
    n_layers: int
    dim: int
    n_heads: int
    output_size: int
    max_batch_size: int
    max_seq_len: int
    an_object_is_in_hand: bool = True
    action_space: SPOCV1ActionSpace = SPOCV1ActionSpace()
    dropout: float = 0.1

    @property
    def decoder(self) -> LlamaConfig:
        return LlamaConfig(
            n_layers=self.n_layers,
            dim=self.dim,
            n_heads=self.n_heads,
            output_size=self.output_size,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
        )


class LlamaActionDecoder(nn.Module):
    def __init__(
        self,
        cfg: LlamaActionDecoderConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.decoder = LlamaDecoder(cfg.decoder)

        self.time_encoder = PositionalEncoder(self.cfg.decoder.dim, self.cfg.decoder.max_seq_len)

        # if num_actions=20; then 0-19 are actions, 20 is for padding, and 21 is for "" (start token)
        self.last_actions_embed = nn.Embedding(
            self.cfg.action_space.get_num_actions() + 2,
            self.cfg.decoder.dim,
            padding_idx=self.cfg.action_space.get_num_actions(),
        )
        self.last_actions_embed.weight.data.uniform_(-0.01, 0.01)

        if cfg.an_object_is_in_hand:
            self.object_in_hand_embed = nn.Embedding(3, self.cfg.decoder.dim)
            self.object_in_hand_embed.weight.data.uniform_(-0.01, 0.01)

    def get_input_embedding_per_timestep(
        self,
        goal_cond_visual_feats: torch.Tensor,  # B, T, D
        last_actions: torch.Tensor,  # B, T
        time_ids: torch.Tensor,  # B, T
        an_object_is_in_hand: Optional[torch.Tensor] = None,  # B, T
    ) -> torch.Tensor:

        last_actions_enc = self.last_actions_embed(last_actions)
        state_embed = goal_cond_visual_feats + last_actions_enc

        if an_object_is_in_hand is not None:
            object_in_hand_enc = self.object_in_hand_embed(an_object_is_in_hand)
            state_embed = state_embed + object_in_hand_enc

        time_enc = self.time_encoder(time_ids)
        state_embed = state_embed + time_enc

        return state_embed

    def decode(
        self,
        state_embed,
        text_feats=None,
        causal_mask=None,
        kv_cache_step_counter=None,
        padding_mask=None,
    ):
        decoder_output = self.decoder(
            tokens=state_embed,
            start_pos=kv_cache_step_counter,
            mask=causal_mask,
        )
        return decoder_output

    def forward(
        self,
        goal_cond_visual_feats: torch.Tensor,  # B, T, D
        goals_features: torch.Tensor,  # B, L, D
        last_actions: torch.Tensor,  # B, T
        time_ids: torch.Tensor,  # B, T
        causal_mask: Optional[torch.Tensor] = None,  # B, T, T
        padding_mask: Optional[torch.Tensor] = None,  # B, T
        an_object_is_in_hand: Optional[torch.Tensor] = None,  # B, T
        kv_cache_step_counter: Optional[int] = None,
    ):
        state_embed = self.get_input_embedding_per_timestep(
            goal_cond_visual_feats,
            last_actions,
            time_ids,
            an_object_is_in_hand,
        )

        return self.decode(
            state_embed=state_embed,
            text_feats=goals_features,
            kv_cache_step_counter=kv_cache_step_counter,
            causal_mask=causal_mask,
            padding_mask=padding_mask,
        )


@dataclass
class GoalCondLlamaActionDecoderConfig:
    n_layers: int
    dim: int
    n_heads: int
    output_size: int
    max_batch_size: int
    max_seq_len: int
    an_object_is_in_hand: bool = True
    action_space: SPOCV1ActionSpace = SPOCV1ActionSpace()
    dropout: float = 0.1
    use_rms_norm: bool = True
    norm_first: bool = True
    activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = nn.functional.silu

    @property
    def decoder(self) -> LlamaConfig:
        return GoalCondLlamaConfig(
            n_layers=self.n_layers,
            dim=self.dim,
            n_heads=self.n_heads,
            output_size=self.output_size,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            use_rms_norm=self.use_rms_norm,
            norm_first=self.norm_first,
            activation=self.activation,
        )


class GoalCondLlamaActionDecoder(LlamaActionDecoder):
    def __init__(
        self,
        cfg: GoalCondLlamaActionDecoderConfig,
    ):
        super().__init__(cfg)
        self.decoder = GoalCondLlamaDecoder(cfg.decoder)

    def decode(
        self,
        state_embed,
        text_feats=None,
        causal_mask=None,
        kv_cache_step_counter=None,
        padding_mask=None,
    ):
        decoder_output = self.decoder(
            tokens=state_embed,
            memory=text_feats,
            start_pos=kv_cache_step_counter,
            mask=causal_mask,
            padding_mask=padding_mask,
        )
        return decoder_output
