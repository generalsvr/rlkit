from __future__ import annotations

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .features import make_features


def compute_rl_signals(df: pd.DataFrame, model_path: str, window: int = 128) -> pd.DataFrame:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    device = os.environ.get("RL_DEVICE", "cuda")
    model = PPO.load(model_path, device=device)

    # Optional: load VecNormalize stats for observation normalization
    obs_mean = None
    obs_var = None
    clip_obs = 10.0
    stats_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if os.path.exists(stats_path):
        try:
            import cloudpickle  # type: ignore
            with open(stats_path, "rb") as f:
                vec = cloudpickle.load(f)
            # Expecting a VecNormalize instance
            if hasattr(vec, "obs_rms") and hasattr(vec.obs_rms, "mean") and hasattr(vec.obs_rms, "var"):
                obs_mean = np.asarray(vec.obs_rms.mean, dtype=np.float32)
                obs_var = np.asarray(vec.obs_rms.var, dtype=np.float32)
                clip_obs = float(getattr(vec, "clip_obs", 10.0))
        except Exception:
            pass

    def _apply_norm(obs: np.ndarray) -> np.ndarray:
        if obs_mean is None or obs_var is None:
            return obs
        # Guard against shape mismatch between saved VecNormalize stats and current obs
        if obs.shape[-1] != int(np.size(obs_mean)):
            return obs
        return np.clip((obs - obs_mean) / (np.sqrt(obs_var + 1e-8)), -clip_obs, clip_obs)

    # Enforce training feature layout if available
    # Prefer model-specific feature columns file; fallback to generic
    base = os.path.splitext(os.path.basename(model_path))[0]
    feat_cols_path_model = os.path.join(os.path.dirname(model_path), f"{base}.feature_columns.json")
    feat_cols_path_generic = os.path.join(os.path.dirname(model_path), "feature_columns.json")
    feat_cols_path = feat_cols_path_model if os.path.exists(feat_cols_path_model) else feat_cols_path_generic
    feature_columns = None
    if os.path.exists(feat_cols_path):
        try:
            import json as _json
            with open(feat_cols_path, "r") as f:
                feature_columns = _json.load(f)
        except Exception:
            feature_columns = None
    # Auto-detect feature mode and higher TFs from saved columns
    mode_for_eval = None
    extra_timeframes = None
    if isinstance(feature_columns, (list, tuple)):
        # Basic mode detector
        if any(col in feature_columns for col in ("close_z", "change", "d_hl")):
            mode_for_eval = "basic"
        # Extract HTF prefixes like '4H', '1D'
        tfs = set()
        for col in feature_columns:
            if isinstance(col, str) and "_" in col:
                prefix = col.split("_", 1)[0]
                s = prefix.strip().upper()
                # crude timeframe pattern: digits + H or D
                if len(s) >= 2 and (s.endswith("H") or s.endswith("D")):
                    head = s[:-1]
                    if head.isdigit():
                        tfs.add(s)
        if tfs:
            extra_timeframes = sorted(tfs)
    feats = make_features(df, feature_columns=feature_columns, mode=mode_for_eval, extra_timeframes=extra_timeframes)
    feats = feats.reset_index(drop=True)
    n = len(feats)
    actions = np.zeros(n, dtype=int)

    # Rolling inference, causal, with position fed back into observation
    position = 0
    enter_long = np.zeros(n, dtype=int)
    exit_long = np.zeros(n, dtype=int)
    enter_short = np.zeros(n, dtype=int)
    exit_short = np.zeros(n, dtype=int)
    deterministic_flag = os.environ.get("RL_DETERMINISTIC", "true").lower() in ("1", "true", "yes")

    for t in range(window, n):
        window_slice = feats.iloc[t - window:t].values.astype(np.float32)
        pos_feat = np.full((window, 1), float(position), dtype=np.float32)
        obs = np.concatenate([window_slice, pos_feat], axis=1).reshape(1, -1)
        obs = _apply_norm(obs)
        a, _ = model.predict(obs, deterministic=deterministic_flag)
        act = int(a[0])
        actions[t] = act
        if position == 0:
            if act == 1:  # take_long
                enter_long[t] = 1
                position = 1
            elif act == 2:  # take_short
                enter_short[t] = 1
                position = -1
        elif position == 1:
            if act == 3:
                exit_long[t] = 1
                position = 0
            elif act == 2:
                # Flip: long -> short
                exit_long[t] = 1
                enter_short[t] = 1
                position = -1
        elif position == -1:
            if act == 3:
                exit_short[t] = 1
                position = 0
            elif act == 1:
                # Flip: short -> long
                exit_short[t] = 1
                enter_long[t] = 1
                position = 1

    out = feats.copy()
    out["rl_action"] = actions
    # Backward-compatible columns
    out["rl_buy"] = enter_long
    out["rl_sell"] = (exit_long + exit_short).clip(0, 1)
    # Preferred modern interface for Freqtrade
    out["enter_long"] = enter_long
    out["exit_long"] = exit_long
    out["enter_short"] = enter_short
    out["exit_short"] = exit_short
    return out


