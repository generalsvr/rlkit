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
    feat_cols_path = os.path.join(os.path.dirname(model_path), "feature_columns.json")
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
    # Align feature dimension to model's expected observation size if needed
    try:
        expected_flat = int(getattr(getattr(model, "observation_space", None) or getattr(model.policy, "observation_space", None), "shape", [0])[0])
    except Exception:
        expected_flat = 0
    if expected_flat and window > 0:
        # model expects: (window * (num_features + 1)) where +1 is position channel
        target_feat_dim = max(1, (expected_flat // int(window)) - 1)
        current_feat_dim = int(feats.shape[1])
        if current_feat_dim != target_feat_dim:
            if current_feat_dim > target_feat_dim:
                # Trim extra columns from the right (HTFs were appended after base)
                feats = feats.iloc[:, :target_feat_dim].copy()
            else:
                # Pad with zeros for missing columns to match expected size
                for i in range(current_feat_dim, target_feat_dim):
                    feats[f"pad_{i}"] = 0.0
                # Ensure exact ordering
                feats = feats.iloc[:, :target_feat_dim].copy()
    feats = feats.reset_index(drop=True)
    n = len(feats)
    actions = np.zeros(n, dtype=int)

    # Rolling inference with next-open execution parity and gating
    position = 0  # effective position after last executed change
    pending_target: int | None = None
    last_change_idx = -10**9
    cooldown_until_idx = -1
    # Gates via env vars (optional)
    try:
        min_hold_bars = int(os.environ.get("RL_MIN_HOLD_BARS", "0"))
    except Exception:
        min_hold_bars = 0
    try:
        cooldown_bars = int(os.environ.get("RL_COOLDOWN_BARS", "0"))
    except Exception:
        cooldown_bars = 0

    enter_long = np.zeros(n, dtype=int)
    exit_long = np.zeros(n, dtype=int)
    enter_short = np.zeros(n, dtype=int)
    exit_short = np.zeros(n, dtype=int)
    deterministic_flag = os.environ.get("RL_DETERMINISTIC", "true").lower() in ("1", "true", "yes")

    for t in range(window, n):
        # Execute any pending target at this bar's open (signals set on this bar)
        if pending_target is not None:
            # Apply min-hold/cooldown gating already enforced at scheduling time
            if pending_target == 0 and position != 0:
                if position == 1:
                    exit_long[t] = 1
                elif position == -1:
                    exit_short[t] = 1
                position = 0
                last_change_idx = t
                if cooldown_bars > 0:
                    cooldown_until_idx = t + cooldown_bars
            elif pending_target == 1 and position <= 0:
                if position == -1:
                    exit_short[t] = 1
                enter_long[t] = 1
                position = 1
                last_change_idx = t
            elif pending_target == -1 and position >= 0:
                if position == 1:
                    exit_long[t] = 1
                enter_short[t] = 1
                position = -1
                last_change_idx = t
            pending_target = None

        # Build observation with current effective position
        window_slice = feats.iloc[t - window:t].values.astype(np.float32)
        pos_feat = np.full((window, 1), float(position), dtype=np.float32)
        obs = np.concatenate([window_slice, pos_feat], axis=1).reshape(1, -1)
        obs = _apply_norm(obs)
        a, _ = model.predict(obs, deterministic=deterministic_flag)
        act = int(a[0])
        actions[t] = act

        # Decide desired target to be executed at t+1, subject to gating
        if t + 1 >= n:
            break
        can_close = (t - last_change_idx) >= max(0, min_hold_bars)
        can_open = t >= cooldown_until_idx
        desired: int | None = None
        if act == 1:  # request long
            if position <= 0 and can_open:
                desired = +1
        elif act == 2:  # request short
            if position >= 0 and can_open:
                desired = -1
        elif act == 3:  # request close
            if position != 0 and can_close:
                desired = 0
        # Schedule for next bar
        if desired is not None and pending_target is None:
            pending_target = desired

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


