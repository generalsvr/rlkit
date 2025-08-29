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

    out = feats.copy()
    # Map actions to buy/sell for spot-only trading
    out["rl_action"] = actions
    out["rl_buy"] = (actions == 1).astype(int)
    out["rl_sell"] = ((actions == 2) | (actions == 3)).astype(int)
    return out


