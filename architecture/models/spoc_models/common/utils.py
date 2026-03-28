from dataclasses import dataclass
import math
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    n_layers: int = 3
    dim: int = 512
    n_heads: int = 8
    batch_first: bool = True


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.dim, nhead=cfg.n_heads, batch_first=cfg.batch_first
            ),
            num_layers=cfg.n_layers,
        )

    def forward(self, x):
        return self.encoder(x)


class TransformerDecoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.cfg.dim,
                nhead=self.cfg.n_heads,
                batch_first=cfg.batch_first,
            ),
            num_layers=self.cfg.n_layers,
        )

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        return self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )


class PositionalEncoder(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        self.register_buffer("div_term", div_term)

    def forward(self, position):
        """
        Args:
            position: Tensor, shape [batch_size, seq_len]
        """
        B, L = position.shape
        position = position.unsqueeze(-1)  # BxLx1
        pe = torch.zeros([B, L, self.dim], device=position.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe


class LinearActorHead(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        # Disable to debug
        # nn.init.orthogonal_(self.linear.weight, gain=0.01)
        # nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x)  # type:ignore

        # noinspection PyArgumentList
        return x  # logits are [step, sampler, ...]


class LinearCriticHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x).view(*x.shape[:2], -1)  # [steps, samplers, flattened]


class SimpleSumFusionModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[1] > 1, "Need at least two inputs to sum"
        y = x.sum(1).unsqueeze(1)
        x = torch.cat([y, x], dim=1)
        return x
