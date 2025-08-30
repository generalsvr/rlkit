from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

# Reuse project libs
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rl_lib.train_sb3 import _find_data_file, _load_ohlcv  # type: ignore
from rl_lib.features import make_features  # type: ignore
from rl_lib.env import FTTradingEnv, TradingConfig  # type: ignore
from stable_baselines3 import PPO  # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore
from stable_baselines3.common.monitor import Monitor  # type: ignore


app = FastAPI(title="RL Backtest Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CandlesQuery(BaseModel):
    pair: str
    timeframe: str = "1h"
    userdir: str = str(ROOT / "freqtrade_userdir")
    exchange: Optional[str] | None = "bybit"
    timerange: Optional[str] | None = None


class RunQuery(BaseModel):
    pair: str
    timeframe: str = "1h"
    userdir: str = str(ROOT / "freqtrade_userdir")
    model_path: str = str(ROOT / "models" / "rl_ppo.zip")
    exchange: Optional[str] | None = "bybit"
    timerange: Optional[str] | None = None
    window: int = 128
    fee_bps: float = 6.0
    slippage_bps: float = 2.0
    idle_penalty_bps: float = 0.02
    reward_type: str = "vol_scaled"
    vol_lookback: int = 20
    turnover_penalty_bps: float = 0.0
    dd_penalty: float = 0.0
    min_hold_bars: int = 0
    cooldown_bars: int = 0
    episode_max_steps: int = 0
    feature_mode: str = "full"
    basic_lookback: int = 64


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/candles")
def candles(q: CandlesQuery) -> JSONResponse:
    path = _find_data_file(q.userdir, q.pair, q.timeframe, prefer_exchange=q.exchange)
    if not path:
        raise HTTPException(status_code=404, detail="Data not found")
    df = _load_ohlcv(path)
    if q.timerange:
        try:
            start_str, end_str = q.timerange.split("-", 1)
            start = pd.to_datetime(start_str) if start_str else None
            end = pd.to_datetime(end_str) if end_str else None
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index
                try:
                    idx_cmp = idx.tz_convert(None) if idx.tz is not None else idx
                except Exception:
                    idx_cmp = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
                mask = pd.Series(True, index=idx)
                if start is not None:
                    mask &= idx_cmp >= pd.to_datetime(start)
                if end is not None:
                    mask &= idx_cmp <= pd.to_datetime(end)
                df = df.loc[mask]
        except Exception:
            pass
    # Return OHLCV array for charting
    out = []
    idx = df.index
    for i in range(len(df)):
        ts = int(pd.Timestamp(idx[i]).value // 10**6) if isinstance(idx, pd.DatetimeIndex) else int(i)
        row = df.iloc[i]
        out.append([ts, float(row.get("open", np.nan)), float(row.get("high", np.nan)), float(row.get("low", np.nan)), float(row.get("close", np.nan)), float(row.get("volume", 0.0))])
    # Filter out rows missing OHLC to avoid client errors
    filtered = [c for c in out if all(np.isfinite(v) for v in c[1:5])]
    return JSONResponse({"candles": filtered})


def _slice_timerange(df: pd.DataFrame, timerange: Optional[str]) -> pd.DataFrame:
    if not timerange:
        return df
    try:
        start_str, end_str = timerange.split("-", 1)
        start = pd.to_datetime(start_str) if start_str else None
        end = pd.to_datetime(end_str) if end_str else None
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            try:
                idx_cmp = idx.tz_convert(None) if idx.tz is not None else idx
            except Exception:
                idx_cmp = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
            mask = pd.Series(True, index=idx)
            if start is not None:
                mask &= idx_cmp >= pd.to_datetime(start)
            if end is not None:
                mask &= idx_cmp <= pd.to_datetime(end)
            return df.loc[mask]
    except Exception:
        return df
    return df


@app.post("/run")
def run_agent(q: RunQuery) -> JSONResponse:
    path = _find_data_file(q.userdir, q.pair, q.timeframe, prefer_exchange=q.exchange)
    if not path:
        raise HTTPException(status_code=404, detail="Data not found")
    raw = _load_ohlcv(path)
    raw = _slice_timerange(raw, q.timerange)

    # Try to enforce training feature layout for model consistency
    feature_columns = None
    try:
        import json as _json
        feat_cols_path = os.path.join(os.path.dirname(q.model_path), "feature_columns.json")
        if os.path.exists(feat_cols_path):
            with open(feat_cols_path, "r") as f:
                feature_columns = _json.load(f)
    except Exception:
        feature_columns = None

    # Auto-detect mode from saved columns
    mode_for_eval = q.feature_mode
    if isinstance(feature_columns, (list, tuple)):
        if any(col in feature_columns for col in ("close_z", "change", "d_hl")):
            mode_for_eval = "basic"

    feats = make_features(raw, feature_columns=feature_columns, mode=mode_for_eval, basic_lookback=q.basic_lookback)

    tcfg = TradingConfig(
        window=q.window,
        fee_bps=q.fee_bps,
        slippage_bps=q.slippage_bps,
        reward_scale=1.0,
        pnl_on_close=False,
        idle_penalty_bps=q.idle_penalty_bps,
        reward_type=q.reward_type,
        vol_lookback=q.vol_lookback,
        turnover_penalty_bps=q.turnover_penalty_bps,
        dd_penalty=q.dd_penalty,
        min_hold_bars=q.min_hold_bars,
        cooldown_bars=q.cooldown_bars,
        random_reset=False,
        episode_max_steps=q.episode_max_steps,
    )

    def make_eval():
        env = FTTradingEnv(feats.copy(), tcfg)
        return Monitor(env)

    eval_env = DummyVecEnv([make_eval])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Inject saved VecNormalize stats if available
    stats_path = os.path.join(os.path.dirname(q.model_path), "vecnormalize.pkl")
    if os.path.exists(stats_path):
        try:
            import cloudpickle  # type: ignore
            with open(stats_path, "rb") as f:
                vec = cloudpickle.load(f)
            if hasattr(vec, "obs_rms") and hasattr(eval_env, "obs_rms"):
                eval_env.obs_rms = vec.obs_rms
        except Exception:
            pass

    model = PPO.load(q.model_path, device=os.environ.get("RL_DEVICE", "cuda"))

    # Rollout and collect actions/positions
    obs = eval_env.reset()
    actions: List[int] = []
    positions: List[int] = []
    equities: List[float] = []
    logs: List[Dict[str, Any]] = []

    step = 0
    while True:
        a, _ = model.predict(obs, deterministic=True)
        action = int(a[0] if isinstance(a, (list, np.ndarray)) else a)
        obs, reward, dones, infos = eval_env.step([action])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        pos = int(info.get("position", 0))
        eq = float(info.get("equity", np.nan))
        actions.append(action)
        positions.append(pos)
        equities.append(eq)
        logs.append({"step": step, "action": action, "position": pos, "equity": eq, "reward": float(reward[0] if isinstance(reward, (list, np.ndarray)) else reward)})
        step += 1
        done_flag = bool(dones[0]) if isinstance(dones, (list, np.ndarray)) else bool(dones)
        if done_flag:
            break

    # Build marks for entries/exits
    marks: List[Dict[str, Any]] = []
    # Align timestamps to base data index after warmup window
    index = feats.index
    base_ptr_start = tcfg.window
    for i in range(1, len(positions)):
        prev = positions[i - 1]
        curr = positions[i]
        if prev == curr:
            continue
        idx_pos = base_ptr_start + i
        if idx_pos >= len(index):
            break
        ts = int(pd.Timestamp(index[idx_pos]).value // 10**6) if isinstance(index, pd.DatetimeIndex) else int(idx_pos)
        label = ""
        color = "#aaa"
        if prev == 0 and curr == 1:
            label = "ENTER_LONG"
            color = "#16a34a"
        elif prev == 0 and curr == -1:
            label = "ENTER_SHORT"
            color = "#dc2626"
        elif prev == 1 and curr == 0:
            label = "EXIT_LONG"
            color = "#0ea5e9"
        elif prev == -1 and curr == 0:
            label = "EXIT_SHORT"
            color = "#0ea5e9"
        marks.append({"time": ts, "label": label, "color": color})

    # Return data for charting
    candles = []
    base = feats[["open", "high", "low", "close", "volume"]].copy()
    for i in range(len(base)):
        ts = int(pd.Timestamp(base.index[i]).value // 10**6) if isinstance(base.index, pd.DatetimeIndex) else int(i)
        r = base.iloc[i]
        candles.append([ts, float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r.get("volume", 0.0))])

    return JSONResponse({
        "candles": candles,
        "marks": marks,
        "equity": equities,
        "actions": actions,
        "positions": positions,
        "logs": logs,
    })


def start():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    start()


