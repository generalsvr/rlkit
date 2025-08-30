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
    idle_penalty_bps: float = 0.0
    # Reward shaping and risk controls
    reward_type: str = "raw"  # raw | vol_scaled | sharpe_proxy
    vol_lookback: int = 20
    turnover_penalty_bps: float = 0.0
    dd_penalty: float = 0.0
    min_hold_bars: int = 0
    cooldown_bars: int = 0
    # Training ergonomics
    random_reset: bool = False
    episode_max_steps: int = 0  # 0 means run to end of dataset


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
        self.episode_steps = 0
        self.last_pos_change_step = -10**9
        self.cooldown_until_step = -1
        self.max_equity_seen = 1.0

        # Price series for PnL
        if "close" not in self.df.columns:
            raise ValueError("features must include 'close' column for PnL calc")
        self.close = self.df["close"].astype(float).values
        # Precompute simple returns and rolling volatility for vol-scaled rewards
        close_shift = np.roll(self.close, 1)
        close_shift[0] = self.close[0]
        self.simple_ret = (self.close - close_shift) / (close_shift + 1e-12)
        try:
            ret_std = pd.Series(self.simple_ret).rolling(self.cfg.vol_lookback, min_periods=1).std().fillna(0.0).to_numpy()
        except Exception:
            ret_std = np.zeros_like(self.simple_ret)
        self.ret_std = np.maximum(ret_std, 1e-8)

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

    def _get_risk_gate(self, idx: int) -> float:
        try:
            if "risk_gate" in self.df.columns and 0 <= idx < len(self.df):
                g = float(self.df.at[idx, "risk_gate"])  # type: ignore[index]
                if not np.isfinite(g):
                    return 1.0
                return float(np.clip(g, 0.0, 1.0))
        except Exception:
            pass
        return 1.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Randomized start to improve sample efficiency if enabled
        self.episode_steps = 0
        self.last_pos_change_step = -10**9
        self.cooldown_until_step = -1
        n = len(self.close)
        if self.cfg.random_reset and self.cfg.episode_max_steps > 0 and n > (self.cfg.window + self.cfg.episode_max_steps + 1):
            low = self.cfg.window
            high = n - self.cfg.episode_max_steps
            # Avoid degenerate ranges
            start = int(np.random.randint(low, max(low + 1, high)))
            self.ptr = start
        else:
            self.ptr = self.cfg.window
        self.done = False
        self.position = 0
        self.entry_price = None
        self.equity = 1.0
        self.equity_curve = [self.equity]
        self.max_equity_seen = self.equity
        obs = self._obs()
        return obs, {}

    def _obs(self):
        w = self.df.iloc[self.ptr - self.cfg.window:self.ptr][self.feature_cols].values.astype(np.float32)
        # Replace NaN/Inf values defensively to avoid propagating invalid tensors
        w = np.nan_to_num(w, nan=0.0, posinf=1e6, neginf=-1e6)
        pos = np.full((self.cfg.window, 1), float(self.position), dtype=np.float32)
        x = np.concatenate([w, pos], axis=1).astype(np.float32)
        return x.reshape(-1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.done:
            raise RuntimeError("Step called on done env")

        prev_price = float(self.close[self.ptr - 1])
        prev_position = self.position
        step_penalty = 0.0
        action_blocked = False

        # apply action
        can_close = (self.episode_steps - self.last_pos_change_step) >= self.cfg.min_hold_bars
        can_open = self.episode_steps >= self.cooldown_until_step
        gate_at_exec = self._get_risk_gate(self.ptr - 1)
        if action == 1:  # long
            if self.position <= 0 and can_open:
                if self.position != +1:
                    cost = self._pay_costs(prev_position, +1, size_factor=gate_at_exec)
                    step_penalty += cost
                self.position = +1
                self.entry_price = prev_price
                self.last_pos_change_step = self.episode_steps
            else:
                action_blocked = True
        elif action == 2:  # short
            if self.position >= 0 and can_open:
                if self.position != -1:
                    cost = self._pay_costs(prev_position, -1, size_factor=gate_at_exec)
                    step_penalty += cost
                self.position = -1
                self.entry_price = prev_price
                self.last_pos_change_step = self.episode_steps
            else:
                action_blocked = True
        elif action == 3:  # close
            if self.position != 0 and can_close:
                cost = self._pay_costs(prev_position, 0, size_factor=gate_at_exec)
                step_penalty += cost
                if self.cfg.pnl_on_close and self.entry_price is not None:
                    r = (prev_price - self.entry_price) / (self.entry_price + 1e-12)
                    r = r if prev_position == +1 else -r
                    # Apply current gate to PnL on close as approximation
                    g = self._get_risk_gate(self.ptr - 1)
                    self.equity *= (1.0 + g * r)
                self.position = 0
                self.entry_price = None
                self.last_pos_change_step = self.episode_steps
                if self.cfg.cooldown_bars > 0:
                    self.cooldown_until_step = self.episode_steps + self.cfg.cooldown_bars
            else:
                action_blocked = True

        # advance
        self.ptr += 1
        self.episode_steps += 1
        if self.ptr >= len(self.close):
            self.done = True
        if (not self.done) and self.cfg.episode_max_steps > 0 and self.episode_steps >= self.cfg.episode_max_steps:
            self.done = True

        reward = 0.0
        if not self.cfg.pnl_on_close and not self.done:
            new_price = float(self.close[self.ptr - 1])
            r = (new_price - prev_price) / (prev_price + 1e-12)
            pos_sign = 1.0 if self.position == +1 else (-1.0 if self.position == -1 else 0.0)
            g = self._get_risk_gate(self.ptr - 1)
            raw_ret = (pos_sign * g) * r
            # Apply reward shaping
            if self.cfg.reward_type == "vol_scaled":
                # Use precomputed rolling volatility at prev index
                sigma = float(self.ret_std[self.ptr - 1])
                reward = raw_ret / (sigma + 1e-8)
            elif self.cfg.reward_type == "sharpe_proxy":
                sigma = float(self.ret_std[self.ptr - 1])
                reward = raw_ret / (sigma + 1e-8)
                # Drawdown penalty term
                dd = max(0.0, (self.max_equity_seen - self.equity) / (self.max_equity_seen + 1e-12))
                reward -= float(self.cfg.dd_penalty) * dd
            else:
                reward = raw_ret
            # Update equity by raw PnL
            if self.position == +1:
                self.equity *= (1.0 + g * r)
            elif self.position == -1:
                self.equity *= (1.0 - g * r)
            else:
                # Encourage taking positions by applying a tiny idle penalty (in bps)
                if self.cfg.idle_penalty_bps > 0.0:
                    idle_cost = self.cfg.idle_penalty_bps * 1e-4
                    reward -= idle_cost
                    self.equity *= (1.0 - idle_cost)
        # Turnover penalty (separate from exchange fees) when we attempted and executed a change
        if not action_blocked and self.cfg.turnover_penalty_bps > 0.0 and self.episode_steps > 0:
            # Charge only when a position change occurred this step
            if prev_position != self.position:
                g = self._get_risk_gate(self.ptr - 1)
                tcost = (self.cfg.turnover_penalty_bps * 1e-4) * g
                step_penalty += tcost
                self.equity *= (1.0 - tcost)

        self.equity_curve.append(self.equity)
        self.max_equity_seen = max(self.max_equity_seen, self.equity)
        obs = self._obs() if not self.done else np.zeros_like(self._obs(), dtype=np.float32)
        terminated = self.done
        truncated = False
        # Subtract exchange/slippage penalties from reward
        reward = reward - step_penalty
        info = {"equity": float(self.equity), "position": int(self.position), "risk_gate": float(self._get_risk_gate(self.ptr - 1))}
        return obs, float(reward * self.cfg.reward_scale), terminated, truncated, info

    def _pay_costs(self, from_pos: int, to_pos: int, size_factor: float = 1.0) -> float:
        if from_pos == to_pos:
            return 0.0
        bps = self.cfg.fee_bps + self.cfg.slippage_bps
        # Flip (long<->short) incurs exit+entry costs
        if from_pos * to_pos == -1:
            bps *= 2.0
        # Scale fees/slippage by current effective size
        size = float(np.clip(size_factor, 0.0, 1.0))
        cost = (bps * 1e-4) * size
        self.equity *= (1.0 - cost)
        return float(cost)


