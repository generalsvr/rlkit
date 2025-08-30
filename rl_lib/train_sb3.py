from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from .features import make_features
from .transformer_extractor import TransformerExtractor
from .env import FTTradingEnv, TradingConfig


def _find_data_file(userdir: str, pair: str, timeframe: str) -> Optional[str]:
    # Support multiple naming variants across exchanges (e.g., Bybit futures)
    base = pair.replace("/", "_")  # BTC_USDT or BTC_USDT:USDT
    candidates = {
        base,
        base.replace(":", "_"),          # BTC_USDT_USDT
        base.split(":")[0],               # BTC_USDT
    }
    for ext in ("parquet", "feather"):
        for name in candidates:
            pattern = os.path.join(userdir, "data", "**", f"{name}-{timeframe}.{ext}")
            hits = glob.glob(pattern, recursive=True)
            if hits:
                return hits[0]
    return None


def _load_ohlcv(path: str) -> pd.DataFrame:
    if path.endswith(".feather"):
        df = pd.read_feather(path)
    else:
        df = pd.read_parquet(path)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    # Freqtrade parquet typically contains: date, open, high, low, close, volume
    time_col = cols.get("date") or cols.get("time") or cols.get("datetime")
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).set_index(time_col)

    # Optional: merge Bybit funding rate (8h) if available
    try:
        base = os.path.basename(path)
        name_noext = os.path.splitext(base)[0]
        # e.g., BTC_USDT_USDT-1h -> pair_key = BTC_USDT_USDT
        pair_key = name_noext.rsplit('-', 1)[0]
        exchange_dir = os.path.dirname(path)
        funding_path = os.path.join(os.path.dirname(exchange_dir), os.path.basename(exchange_dir), "futures", f"{pair_key}-8h-funding_rate.parquet")
        # Fallback: typical layout /data/bybit/futures/<file>
        if not os.path.exists(funding_path):
            funding_path = os.path.join(os.path.dirname(exchange_dir), "futures", f"{pair_key}-8h-funding_rate.parquet")
        if os.path.exists(funding_path):
            fr = pd.read_parquet(funding_path)
            c2 = {c.lower(): c for c in fr.columns}
            tcol = c2.get("date") or c2.get("time") or c2.get("datetime")
            if tcol is not None:
                fr[tcol] = pd.to_datetime(fr[tcol])
                fr = fr.sort_values(tcol).set_index(tcol)
            # Expect a column named funding_rate; if not, take first numeric
            if "funding_rate" not in fr.columns:
                num_cols = [c for c in fr.columns if pd.api.types.is_numeric_dtype(fr[c])]
                if num_cols:
                    fr = fr.rename(columns={num_cols[0]: "funding_rate"})
            # Resample to 1h, ffill
            fr1h = fr[["funding_rate"]].resample("1H").ffill()
            df = df.join(fr1h, how="left")
            df["funding_rate"] = df["funding_rate"].fillna(0.0).astype(float)
    except Exception:
        pass
    return df


@dataclass
class TrainParams:
    userdir: str
    pair: str
    timeframe: str = "1h"
    window: int = 128
    total_timesteps: int = 200_000
    seed: int = 42
    model_out_path: str = "models/rl_ppo.zip"
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    reward_scale: float = 1.0
    pnl_on_close: bool = False
    arch: str = "mlp"  # mlp | lstm | transformer


def train_ppo_from_freqtrade_data(params: TrainParams) -> str:
    os.makedirs(os.path.dirname(params.model_out_path), exist_ok=True)

    data_path = _find_data_file(params.userdir, params.pair, params.timeframe)
    if not data_path:
        raise FileNotFoundError(
            f"No parquet found for {params.pair} {params.timeframe} under {params.userdir}/data. "
            f"Run data download first."
        )
    raw = _load_ohlcv(data_path)
    feats = make_features(raw)

    # Simple split: 80/20 by time
    n = len(feats)
    cut = int(n * 0.8)
    train_df = feats.iloc[:cut].copy()
    eval_df = feats.iloc[cut - max(params.window, 1):].copy()  # include context

    tcfg = TradingConfig(
        window=params.window,
        fee_bps=params.fee_bps,
        slippage_bps=params.slippage_bps,
        reward_scale=params.reward_scale,
        pnl_on_close=params.pnl_on_close,
    )

    def make_train():
        return FTTradingEnv(train_df, tcfg)

    def make_eval():
        return FTTradingEnv(eval_df, tcfg)

    env = DummyVecEnv([make_train])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = DummyVecEnv([make_eval])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    arch = params.arch.lower()
    if arch == "lstm":
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=10,
            policy_kwargs=dict(lstm_hidden_size=128, net_arch=[128]),
        )
    elif arch == "transformer":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=10,
            ent_coef=0.02,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=TransformerExtractor,
                features_extractor_kwargs=dict(window=params.window, d_model=96, nhead=4, num_layers=2, ff_dim=192),
            ),
        )
    elif arch == "transformer_big":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            n_steps=1024,
            batch_size=128,
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=10,
            ent_coef=0.02,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=TransformerExtractor,
                features_extractor_kwargs=dict(window=params.window, d_model=192, nhead=8, num_layers=4, ff_dim=768),
            ),
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=10,
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(params.model_out_path),
        log_path=os.path.dirname(params.model_out_path),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # Sync eval normalization stats with training env
    if isinstance(env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = env.obs_rms

    model.learn(total_timesteps=int(params.total_timesteps), callback=eval_cb, progress_bar=True)
    model.save(params.model_out_path)
    # Save normalization statistics alongside the model
    if isinstance(env, VecNormalize):
        stats_path = os.path.join(os.path.dirname(params.model_out_path), "vecnormalize.pkl")
        env.save(stats_path)
    return params.model_out_path


