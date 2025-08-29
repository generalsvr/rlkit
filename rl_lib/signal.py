from __future__ import annotations

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .features import make_features


def compute_rl_signals(df: pd.DataFrame, model_path: str, window: int = 128) -> pd.DataFrame:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = PPO.load(model_path, device="cpu")

    feats = make_features(df)
    feats = feats.reset_index(drop=True)
    n = len(feats)
    actions = np.zeros(n, dtype=int)

    # Rolling inference, causal
    for t in range(window, n):
        window_slice = feats.iloc[t - window:t].values.astype(np.float32)
        pos_feat = np.zeros((window, 1), dtype=np.float32)
        obs = np.concatenate([window_slice, pos_feat], axis=1).reshape(1, -1)
        a, _ = model.predict(obs, deterministic=True)
        actions[t] = int(a[0])

    # Derive stateful enter/exit for spot-only trading
    enter_long = np.zeros(n, dtype=int)
    exit_long = np.zeros(n, dtype=int)
    position = 0
    for t in range(window, n):
        act = actions[t]
        if position == 0:
            if act == 1:  # take_long
                enter_long[t] = 1
                position = 1
        elif position == 1:
            if act == 3:  # close_position
                exit_long[t] = 1
                position = 0
            # ignore take_short (act==2) in spot mode

    out = feats.copy()
    out["rl_action"] = actions
    # Backward-compatible columns
    out["rl_buy"] = enter_long
    out["rl_sell"] = exit_long
    # Preferred modern interface for Freqtrade
    out["enter_long"] = enter_long
    out["exit_long"] = exit_long
    return out


