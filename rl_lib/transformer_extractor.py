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


