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
            if hasattr(vec, "obs_rms") and hasattr(vec.obs_rms, "mean") and hasattr(vec.obs_rms, "var"):
                import numpy as _np
                obs_mean = _np.asarray(vec.obs_rms.mean, dtype=_np.float32)
                obs_var = _np.asarray(vec.obs_rms.var, dtype=_np.float32)
                clip_obs = float(getattr(vec, "clip_obs", 10.0))
        except Exception:
            pass

    def _apply_norm(obs):
        if obs_mean is None or obs_var is None:
            return obs
        import numpy as _np
        # Guard against shape mismatch (e.g., different window/features). Fall back to unnormalized.
        if obs.shape[-1] != int(_np.size(obs_mean)):
            return obs
        return _np.clip((obs - obs_mean) / (_np.sqrt(obs_var + 1e-8)), -clip_obs, clip_obs)

    feats = make_features(df)
    feats = feats.reset_index(drop=True)
    n = len(feats)
    actions = np.zeros(n, dtype=int)

    # Rolling inference with position feedback and normalization
    enter_long = np.zeros(n, dtype=int)
    exit_long = np.zeros(n, dtype=int)
    enter_short = np.zeros(n, dtype=int)
    exit_short = np.zeros(n, dtype=int)
    position = 0
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
            if act == 1:
                enter_long[t] = 1
                position = 1
            elif act == 2:
                enter_short[t] = 1
                position = -1
        elif position == 1:
            if act == 3:
                exit_long[t] = 1
                position = 0
        elif position == -1:
            if act == 3:
                exit_short[t] = 1
                position = 0

    out = feats.copy()
    out["rl_action"] = actions
    out["rl_buy"] = enter_long
    out["rl_sell"] = (exit_long + exit_short).clip(0, 1)
    out["enter_long"] = enter_long
    out["exit_long"] = exit_long
    out["enter_short"] = enter_short
    out["exit_short"] = exit_short
    return out


