from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class TradingConfig:
    window: int = 128
    fee_bps: float = 0.6
    slippage_bps: float = 2.0
    reward_scale: float = 1.0
    pnl_on_close: bool = False


class FTTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, config: TradingConfig):
        super().__init__()
        if not isinstance(features.index, (pd.DatetimeIndex, pd.RangeIndex)):
            features = features.reset_index(drop=True)
        self.df = features.reset_index(drop=True).copy()
        self.cfg = config
        self.ptr = self.cfg.window
        self.done = False
        self.position = 0
        self.entry_price = None
        self.equity = 1.0
        self.equity_curve: List[float] = []

        # Price series for PnL
        if "close" not in self.df.columns:
            raise ValueError("features must include 'close' column for PnL calc")
        self.close = self.df["close"].astype(float).values

        # Observation is a flattened window of features + position channel
        self.feature_cols = list(self.df.columns)
        self.obs_feature_dim = len(self.feature_cols) + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_feature_dim * self.cfg.window,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.ptr = self.cfg.window
        self.done = False
        self.position = 0
        self.entry_price = None
        self.equity = 1.0
        self.equity_curve = [self.equity]
        obs = self._obs()
        return obs, {}

    def _obs(self):
        w = self.df.iloc[self.ptr - self.cfg.window:self.ptr][self.feature_cols].values.astype(np.float32)
        pos = np.full((self.cfg.window, 1), float(self.position), dtype=np.float32)
        x = np.concatenate([w, pos], axis=1).astype(np.float32)
        return x.reshape(-1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            raise RuntimeError("Step called on done env")

        prev_price = float(self.close[self.ptr - 1])
        prev_position = self.position

        # apply action
        if action == 1:  # long
            if self.position <= 0:
                self._pay_costs(prev_position, +1)
                self.position = +1
                self.entry_price = prev_price
        elif action == 2:  # short
            if self.position >= 0:
                self._pay_costs(prev_position, -1)
                self.position = -1
                self.entry_price = prev_price
        elif action == 3:  # close
            if self.position != 0:
                self._pay_costs(prev_position, 0)
                if self.cfg.pnl_on_close and self.entry_price is not None:
                    r = (prev_price - self.entry_price) / self.entry_price
                    r = r if prev_position == +1 else -r
                    self.equity *= (1.0 + r)
                self.position = 0
                self.entry_price = None

        # advance
        self.ptr += 1
        if self.ptr >= len(self.close):
            self.done = True

        reward = 0.0
        if not self.cfg.pnl_on_close and not self.done:
            new_price = float(self.close[self.ptr - 1])
            r = (new_price - prev_price) / prev_price
            if self.position == +1:
                reward = r
                self.equity *= (1.0 + r)
            elif self.position == -1:
                reward = -r
                self.equity *= (1.0 - r)

        self.equity_curve.append(self.equity)
        obs = self._obs() if not self.done else np.zeros_like(self._obs(), dtype=np.float32)
        terminated = self.done
        truncated = False
        info = {"equity": float(self.equity), "position": int(self.position)}
        return obs, float(reward * self.cfg.reward_scale), terminated, truncated, info

    def _pay_costs(self, from_pos: int, to_pos: int):
        if from_pos == to_pos:
            return
        bps = self.cfg.fee_bps + self.cfg.slippage_bps
        cost = bps * 1e-4
        self.equity *= (1.0 - cost)


