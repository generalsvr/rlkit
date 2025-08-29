from __future__ import annotations

from typing import Tuple
import math
import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerExtractor(BaseFeaturesExtractor):
    """
    Extractor for flattened (window * features) observation.
    Reshapes to (window, features), applies TransformerEncoder, returns last token.
    """

    def __init__(self, observation_space: spaces.Box, window: int = 128, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, ff_dim: int = 128, dropout: float = 0.1):
        obs_dim = int(observation_space.shape[0])
        assert obs_dim % window == 0, "Observation dim must be divisible by window."
        self.window = window
        self.feature_dim = obs_dim // window

        super().__init__(observation_space, features_dim=d_model)

        self.inp = nn.Linear(self.feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, window*features)
        b, d = observations.shape
        x = observations.view(b, self.window, self.feature_dim)
        h = self.inp(x)
        h = self.pos(h)
        T = h.size(1)
        mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        z = self.encoder(h, mask=mask)
        last = z[:, -1, :]
        return last


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


