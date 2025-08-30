from __future__ import annotations

from typing import Tuple, Dict, List
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
        self.in_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, window*features)
        b, d = observations.shape
        x = observations.view(b, self.window, self.feature_dim)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        h = self.inp(x)
        h = self.in_norm(h)
        h = self.pos(h)
        T = h.size(1)
        mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        z = self.encoder(h, mask=mask)
        last = z[:, -1, :]
        last = self.out_norm(last)
        last = torch.nan_to_num(last, nan=0.0, posinf=1e6, neginf=-1e6)
        return last


class HybridLSTMTransformerExtractor(BaseFeaturesExtractor):
    """
    Hybrid extractor: LSTM full sequence with skip connections into a TransformerEncoder.
    Pipeline:
      (B, T, F) -> LSTM (sequence) and Linear-projected raw -> sum -> LN -> PosEnc -> Transformer -> last token
    """

    def __init__(self, observation_space: spaces.Box, window: int = 128, d_model: int = 96, nhead: int = 4,
                 num_layers: int = 2, ff_dim: int = 192, dropout: float = 0.1, lstm_hidden: int = 128,
                 bidirectional: bool = True):
        obs_dim = int(observation_space.shape[0])
        assert obs_dim % window == 0, "Observation dim must be divisible by window."
        self.window = window
        self.feature_dim = obs_dim // window

        super().__init__(observation_space, features_dim=d_model)

        # Raw input projection path (skip connection)
        self.inp = nn.Linear(self.feature_dim, d_model)

        # LSTM sequence encoder (full sequence output)
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.lstm_proj = nn.Linear(lstm_out_dim, d_model)

        self.in_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, window*features)
        b, d = observations.shape
        x = observations.view(b, self.window, self.feature_dim)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # Parallel paths: raw projection and LSTM sequence features
        h_raw = self.inp(x)
        lstm_out, _ = self.lstm(x)
        h_lstm = self.lstm_proj(lstm_out)

        # Skip connection: combine and normalize
        h = h_raw + h_lstm
        h = self.in_norm(h)
        h = self.pos(h)

        T = h.size(1)
        mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        z = self.encoder(h, mask=mask)
        last = z[:, -1, :]
        last = self.out_norm(last)
        last = torch.nan_to_num(last, nan=0.0, posinf=1e6, neginf=-1e6)
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


class MultiScaleHTFExtractor(BaseFeaturesExtractor):
    """
    Per-HTF encoders with stride downsampling and gated fusion.

    Args:
        window: base window length (in base bars)
        groups: mapping group_name -> list of feature indices (columns within per-step feature vector)
        strides: mapping group_name -> downsample stride in base bars (e.g., base=1, 4H=4, 1D=24)
        d_model: hidden size for each group encoder
        ff_dim, nhead, num_layers, dropout: Transformer encoder hyperparams per group

    Output: fused vector of size d_model
    """

    def __init__(self, observation_space: spaces.Box, window: int, groups: Dict[str, List[int]], strides: Dict[str, int],
                 d_model: int = 128, ff_dim: int = 256, nhead: int = 4, num_layers: int = 1, dropout: float = 0.1):
        obs_dim = int(observation_space.shape[0])
        assert obs_dim % window == 0, "Observation dim must be divisible by window."
        self.window = window
        self.feature_dim = obs_dim // window
        super().__init__(observation_space, features_dim=d_model)

        self.groups = groups
        self.strides = strides
        self.group_names = list(groups.keys())

        # Per-group slicers and encoders
        self.in_linears = nn.ModuleDict()
        self.norms_in = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.norms_out = nn.ModuleDict()
        self.pos = PositionalEncoding(d_model)
        for g, idxs in groups.items():
            in_dim = max(1, len(idxs))
            self.in_linears[g] = nn.Linear(in_dim, d_model)
            self.norms_in[g] = nn.LayerNorm(d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, activation="gelu")
            self.encoders[g] = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.norms_out[g] = nn.LayerNorm(d_model)

        # Learnable gates per group
        self.gates = nn.Parameter(torch.ones(len(self.group_names)))
        self.fuse_norm = nn.LayerNorm(d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, d = observations.shape
        x = observations.view(b, self.window, self.feature_dim)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        fused = None
        gates = torch.softmax(self.gates, dim=0)
        for i, g in enumerate(self.group_names):
            idxs = self.groups[g]
            stride = max(1, int(self.strides.get(g, 1)))
            xs = x[:, :, idxs] if len(idxs) > 1 else x[:, :, idxs[0:1]]
            if stride > 1:
                xs = xs[:, ::stride, :]
            h = self.in_linears[g](xs)
            h = self.norms_in[g](h)
            h = self.pos(h)
            T = h.size(1)
            mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
            z = self.encoders[g](h, mask=mask)
            last = z[:, -1, :]
            last = self.norms_out[g](last)
            last = gates[i] * last
            fused = last if fused is None else fused + last
        fused = self.fuse_norm(fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6)
        return fused


