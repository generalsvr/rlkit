from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .features import make_features
from .transformer_extractor import TransformerExtractor, HybridLSTMTransformerExtractor
from .transformer_extractor import MultiScaleHTFExtractor  # type: ignore
from .env import FTTradingEnv, TradingConfig
from .signal import compute_rl_signals


def _tf_stride(tf: str) -> int:
    """Return stride in base hours for timeframe strings like '4H', '1D'. Defaults to 1 on parse issues."""
    s = str(tf).strip().upper()
    try:
        if s.endswith("H"):
            n = int(s[:-1]) if s[:-1] else 1
            return max(1, n)
        if s.endswith("D"):
            n = int(s[:-1]) if s[:-1] else 1
            return max(1, n) * 24
    except Exception:
        return 1
    return 1


def _find_data_file(userdir: str, pair: str, timeframe: str, prefer_exchange: Optional[str] = None) -> Optional[str]:
    # Support multiple naming variants across exchanges (e.g., Bybit futures)
    base = pair.replace("/", "_")  # BTC_USDT or BTC_USDT:USDT
    candidates = {
        base,
        base.replace(":", "_"),          # BTC_USDT_USDT
        base.split(":")[0],               # BTC_USDT
    }
    # If user typed without futures suffix (e.g., BTC/USDT), still try BTC_USDT_USDT
    up_pair = pair.upper()
    if ":" not in pair and (up_pair.endswith("/USDT") or up_pair.endswith("_USDT")):
        candidates.add(base + "_USDT")     # BTC_USDT_USDT
    suffixes = ["", "-futures"]
    for ext in ("parquet", "feather"):
        for name in candidates:
            for suf in suffixes:
                pattern = os.path.join(userdir, "data", "**", f"{name}-{timeframe}{suf}.{ext}")
                hits = glob.glob(pattern, recursive=True)
                if hits:
                    if prefer_exchange:
                        # Prefer files that contain the exchange segment in their path
                        preferred = [h for h in hits if f"/{prefer_exchange}/" in h]
                        if preferred:
                            return preferred[0]
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
    return df


def _compute_risk_metrics(equities: np.ndarray) -> Dict[str, float]:
    if equities.size < 2:
        return {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown": float("nan"), "calmar": float("nan")}
    rets = np.diff(equities) / (equities[:-1] + 1e-12)
    if rets.size == 0:
        return {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown": 0.0, "calmar": float("nan")}
    mean = float(np.mean(rets))
    std = float(np.std(rets))
    if not np.isfinite(std) or std == 0.0:
        std = 1e-12
    downside = rets[rets < 0.0]
    if downside.size > 0:
        dd_std = float(np.std(downside))
        if not np.isfinite(dd_std) or dd_std == 0.0:
            dd_std = 1e-12
    else:
        dd_std = 1e-12
    cummax = np.maximum.accumulate(equities)
    dd = (cummax - equities) / (cummax + 1e-12)
    max_dd = float(np.max(dd)) if dd.size else 0.0
    sharpe = mean / std
    sortino = mean / dd_std
    calmar = (mean / (max_dd + 1e-12)) if max_dd > 0 else float("nan")
    return {"sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar}


def _run_validation_rollout(model: PPO, eval_env: VecNormalize | DummyVecEnv, max_steps: int = 2000, deterministic: bool = True) -> Dict[str, Any]:
    # Assumes single-env DummyVecEnv wrapped (VecNormalize optional)
    obs = eval_env.reset()
    done = False
    if isinstance(done, np.ndarray):
        done = False
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    enter_long = enter_short = exit_long = exit_short = 0
    positions = []
    equities = []
    steps = 0
    while True:
        a, _ = model.predict(obs, deterministic=deterministic)
        action = int(a[0] if isinstance(a, (list, np.ndarray)) else a)
        action_counts[action] = action_counts.get(action, 0) + 1
        obs, reward, dones, infos = eval_env.step([action])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        pos = int(info.get("position", 0))
        eq = float(info.get("equity", np.nan))
        positions.append(pos)
        equities.append(eq)
        # Entries/exits (0->1, 0->-1, 1->0, -1->0)
        if len(positions) >= 2:
            prev = positions[-2]
            curr = positions[-1]
            if prev == 0 and curr == 1:
                enter_long += 1
            elif prev == 0 and curr == -1:
                enter_short += 1
            elif prev == 1 and curr == 0:
                exit_long += 1
            elif prev == -1 and curr == 0:
                exit_short += 1
        steps += 1
        if isinstance(dones, (list, np.ndarray)):
            if bool(dones[0]):
                break
        else:
            if bool(dones):
                break
        if steps >= max_steps:
            break

    eq_series = np.asarray([e for e in equities if isinstance(e, (int, float)) and not np.isnan(e)], dtype=float)
    final_equity = float(eq_series[-1]) if eq_series.size > 0 else float("nan")
    max_equity = float(np.max(eq_series)) if eq_series.size > 0 else float("nan")
    min_equity = float(np.min(eq_series)) if eq_series.size > 0 else float("nan")
    time_in_pos = float(np.mean(np.asarray(positions) != 0)) if positions else 0.0
    risk = _compute_risk_metrics(eq_series) if eq_series.size > 1 else {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown": float("nan"), "calmar": float("nan")}
    return {
        "steps": steps,
        "action_counts": action_counts,
        "enter_long": enter_long,
        "enter_short": enter_short,
        "exit_long": exit_long,
        "exit_short": exit_short,
        "time_in_position_frac": time_in_pos,
        "final_equity": final_equity,
        "max_equity": max_equity,
        "min_equity": min_equity,
        **risk,
    }


class PnLEvalCallback(BaseCallback):
    def __init__(self, eval_env: VecNormalize | DummyVecEnv, eval_freq: int, max_steps: int = 2000, deterministic: bool = True, verbose: int = 1, eval_log_path: Optional[str] = None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(max(1, eval_freq))
        self.max_steps = int(max_steps)
        self.deterministic = bool(deterministic)
        self.eval_log_path = eval_log_path

    def _append_log(self, num_timesteps: int, report: Dict[str, Any]):
        if not self.eval_log_path:
            return
        path = self.eval_log_path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        fields = [
            "timestamp","timesteps","final_equity","sharpe","max_drawdown","time_in_position_frac",
            "enter_long","enter_short","exit_long","exit_short"
        ]
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "timesteps": int(num_timesteps),
            "final_equity": float(report.get("final_equity", float("nan"))),
            "sharpe": float(report.get("sharpe", float("nan"))),
            "max_drawdown": float(report.get("max_drawdown", float("nan"))),
            "time_in_position_frac": float(report.get("time_in_position_frac", float("nan"))),
            "enter_long": int(report.get("enter_long", 0)),
            "enter_short": int(report.get("enter_short", 0)),
            "exit_long": int(report.get("exit_long", 0)),
            "exit_short": int(report.get("exit_short", 0)),
        }
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _on_step(self) -> bool:
        num = int(self.model.num_timesteps)
        if num % self.eval_freq == 0:
            report = _run_validation_rollout(self.model, self.eval_env, max_steps=self.max_steps, deterministic=self.deterministic)
            final_eq = report.get("final_equity", float("nan"))
            sharpe = report.get("sharpe", float("nan"))
            self.logger.record("eval/final_equity", final_eq)
            self.logger.record("eval/sharpe", sharpe)
            self._append_log(num, report)
            if self.verbose:
                print(f"EVAL_PNL: timesteps={num} final_equity={final_eq:.6f} sharpe={sharpe:.4f}")
        return True


def validate_trained_model(params: TrainParams, max_steps: int = 2000, deterministic: bool = True, timerange: Optional[str] = None) -> Dict[str, Any]:
    # Load data
    data_path = _find_data_file(params.userdir, params.pair, params.timeframe, prefer_exchange=params.prefer_exchange)
    if not data_path:
        raise FileNotFoundError("No data for validation.")
    raw = _load_ohlcv(data_path)
    # Optional restrict to timerange
    if timerange:
        try:
            start_str, end_str = timerange.split('-', 1)
            start = pd.to_datetime(start_str) if start_str else None
            end = pd.to_datetime(end_str) if end_str else None
            if isinstance(raw.index, pd.DatetimeIndex):
                idx = raw.index
                # Normalize to naive for comparison if tz-aware
                try:
                    idx_cmp = idx.tz_convert(None) if idx.tz is not None else idx
                except Exception:
                    idx_cmp = idx.tz_localize(None) if getattr(idx, 'tz', None) is not None else idx
                mask = pd.Series(True, index=idx)
                if start is not None:
                    mask &= idx_cmp >= pd.to_datetime(start)
                if end is not None:
                    mask &= idx_cmp <= pd.to_datetime(end)
                raw = raw.loc[mask]
        except Exception:
            pass
    # Enforce training feature layout if available to match inference/backtest
    feature_columns = None
    try:
        import json as _json
        feat_cols_path = os.path.join(os.path.dirname(params.model_out_path), "feature_columns.json")
        if os.path.exists(feat_cols_path):
            with open(feat_cols_path, "r") as f:
                feature_columns = _json.load(f)
    except Exception:
        feature_columns = None
    # Auto-detect feature mode and HTFs from saved columns
    mode_for_eval = params.feature_mode
    extra_timeframes_for_eval = params.extra_timeframes
    if isinstance(feature_columns, (list, tuple)):
        if any(col in feature_columns for col in ("close_z", "change", "d_hl")):
            mode_for_eval = "basic"
        # Derive HTFs from column prefixes like '4H', '1D'
        tfs = set()
        for col in feature_columns:
            if isinstance(col, str) and "_" in col:
                prefix = col.split("_", 1)[0]
                s = prefix.strip().upper()
                if len(s) >= 2 and (s.endswith("H") or s.endswith("D")):
                    head = s[:-1]
                    if head.isdigit():
                        tfs.add(s)
        if tfs:
            extra_timeframes_for_eval = sorted(tfs)
    feats = make_features(raw, feature_columns=feature_columns, mode=mode_for_eval, basic_lookback=params.basic_lookback, extra_timeframes=extra_timeframes_for_eval)
    eval_df = feats.copy()

    # Diagnostics: show actual validation window after slicing
    if isinstance(eval_df.index, pd.DatetimeIndex) and not eval_df.empty:
        _start_dt = str(eval_df.index[0])
        _end_dt = str(eval_df.index[-1])
        print(f"VALIDATION DATA: rows={len(eval_df)}, start={_start_dt}, end={_end_dt}")
    else:
        print(f"VALIDATION DATA: rows={len(eval_df)} (index type={type(eval_df.index).__name__})")

    tcfg = TradingConfig(
        window=params.window,
        fee_bps=params.fee_bps,
        slippage_bps=params.slippage_bps,
        reward_scale=params.reward_scale,
        pnl_on_close=params.pnl_on_close,
        idle_penalty_bps=params.idle_penalty_bps,
        reward_type=params.reward_type,
        vol_lookback=params.vol_lookback,
        turnover_penalty_bps=params.turnover_penalty_bps,
        dd_penalty=params.dd_penalty,
        min_hold_bars=params.min_hold_bars,
        cooldown_bars=params.cooldown_bars,
        random_reset=params.random_reset,
        episode_max_steps=params.episode_max_steps,
    )

    def make_eval():
        env = FTTradingEnv(eval_df, tcfg)
        return Monitor(env)

    eval_env = DummyVecEnv([make_eval])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Try to load VecNormalize stats if available
    stats_path = os.path.join(os.path.dirname(params.model_out_path), "vecnormalize.pkl")
    if os.path.exists(stats_path):
        try:
            import cloudpickle  # type: ignore
            with open(stats_path, "rb") as f:
                vec = cloudpickle.load(f)
            if hasattr(vec, "obs_rms") and hasattr(eval_env, "obs_rms"):
                eval_env.obs_rms = vec.obs_rms
        except Exception:
            pass

    import os as _os
    model = PPO.load(params.model_out_path, device=_os.environ.get("RL_DEVICE", "cuda"))
    report = _run_validation_rollout(model, eval_env, max_steps=max_steps, deterministic=deterministic)

    # -----------------------------
    # Plot signals with numbering
    # -----------------------------
    try:
        # Ensure gating matches env for signal generation
        os.environ["RL_MIN_HOLD_BARS"] = str(int(params.min_hold_bars))
        os.environ["RL_COOLDOWN_BARS"] = str(int(params.cooldown_bars))
        os.environ["RL_DETERMINISTIC"] = "true" if bool(deterministic) else "false"

        signals_df = compute_rl_signals(eval_df, params.model_out_path, window=int(params.window))
        # Prepare output directory and filename prefix
        outdir = os.path.join(os.path.dirname(params.model_out_path), "validate_plots")
        os.makedirs(outdir, exist_ok=True)
        ts_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"{params.pair.replace('/', '_')}_{params.timeframe}_{ts_tag}".replace(":", "-")
        prefix = os.path.join(outdir, base_name)

        def _plot_signals(df: pd.DataFrame, title: str, save_path: str):
            # X axis
            x = df.index
            # Use close price for plotting
            y = df["close"].astype(float).values if "close" in df.columns else df.iloc[:, 0].astype(float).values

            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(x, y, color="#2c3e50", linewidth=1.0, label="Close")

            # Indices for events
            lo_idx = list(np.flatnonzero(df.get("enter_long", pd.Series(0, index=df.index)).values == 1))
            lc_idx = list(np.flatnonzero(df.get("exit_long", pd.Series(0, index=df.index)).values == 1))
            so_idx = list(np.flatnonzero(df.get("enter_short", pd.Series(0, index=df.index)).values == 1))
            sc_idx = list(np.flatnonzero(df.get("exit_short", pd.Series(0, index=df.index)).values == 1))

            # Pair opens/closes per side to number trades and shade spans
            def pair_trades(open_indices: List[int], close_indices: List[int]) -> List[tuple[int, int | None, int]]:
                open_indices_sorted = sorted(open_indices)
                close_indices_sorted = sorted(close_indices)
                pairs: List[tuple[int, int | None, int]] = []
                active: List[tuple[int, int]] = []  # (open_idx, trade_no)
                trade_no = 0
                # Traverse timeline; process all events in order
                timeline = sorted([(i, 1) for i in open_indices_sorted] + [(i, -1) for i in close_indices_sorted])
                for idx, typ in timeline:
                    if typ == 1:  # open
                        trade_no += 1
                        active.append((idx, trade_no))
                        pairs.append((idx, None, trade_no))
                    else:  # close
                        # match earliest active
                        if active:
                            open_idx, no = active.pop(0)
                            # update pairs for this trade_no
                            for k in range(len(pairs)):
                                if pairs[k][2] == no and pairs[k][1] is None:
                                    pairs[k] = (pairs[k][0], idx, no)
                                    break
                        else:
                            # orphan close, skip pairing
                            pass
                return pairs

            long_pairs = pair_trades(lo_idx, lc_idx)
            short_pairs = pair_trades(so_idx, sc_idx)

            # Shading for positions
            def _shade_spans(pairs: List[tuple[int, int | None, int]], color: str):
                for open_i, close_i, _no in pairs:
                    x0 = x[open_i]
                    x1 = x[close_i] if (close_i is not None and close_i < len(x)) else x[-1]
                    ax.axvspan(x0, x1, color=color, alpha=0.06, linewidth=0)

            _shade_spans(long_pairs, "#2ecc71")
            _shade_spans(short_pairs, "#e74c3c")

            # Plot markers and LO/LC/SO/SC labels with alternating offsets
            def _annotate_pairs(pairs: List[tuple[int, int | None, int]], open_kind: str, close_kind: str, open_color: str, close_color: str, open_marker: str, close_marker: str):
                for open_i, close_i, no in pairs:
                    # Alternate offsets by trade number to reduce overlap
                    open_off = 14 if (no % 2 == 1) else 20
                    close_off = -14 if (no % 2 == 1) else -20
                    # Open
                    if 0 <= open_i < len(y):
                        xi = x[open_i]
                        yi = y[open_i]
                        ax.scatter([xi], [yi], marker=open_marker, color=open_color, s=42, zorder=3)
                        ax.annotate(f"{open_kind}{no}", (xi, yi), textcoords="offset points", xytext=(0, open_off), ha="center",
                                    fontsize=8, color=open_color, fontweight="bold")
                    # Close
                    if close_i is not None and 0 <= close_i < len(y):
                        xi = x[close_i]
                        yi = y[close_i]
                        ax.scatter([xi], [yi], marker=close_marker, color=close_color, s=36, zorder=3)
                        ax.annotate(f"{close_kind}{no}", (xi, yi), textcoords="offset points", xytext=(0, close_off), ha="center",
                                    fontsize=8, color=close_color, fontweight="bold")

            _annotate_pairs(long_pairs, "LO", "LC", open_color="#2ecc71", close_color="#27ae60", open_marker="^", close_marker="o")
            _annotate_pairs(short_pairs, "SO", "SC", open_color="#e74c3c", close_color="#c0392b", open_marker="v", close_marker="o")

            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.25)
            if isinstance(x, pd.DatetimeIndex):
                locator = mdates.AutoDateLocator()
                formatter = mdates.ConciseDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
            ax.legend(["Close"], loc="upper left")
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)

        # Full overview
        _plot_signals(signals_df, title=f"Validation Signals {params.pair} {params.timeframe}", save_path=f"{prefix}_overview.png")
        # Zoomed window (tail)
        zoom_bars = 400
        if len(signals_df) > (zoom_bars + int(params.window)):
            zoom_df = signals_df.iloc[-zoom_bars:].copy()
            _plot_signals(zoom_df, title=f"Validation Signals (Zoom) {params.pair} {params.timeframe}", save_path=f"{prefix}_zoom.png")

        print(f"Saved validation signal plots to: {outdir}")
    except Exception as _e:
        # Do not fail validation if plotting has issues
        print(f"Plotting failed: {_e}")
    # Pretty print
    print("VALIDATION SUMMARY:")
    print(report)
    return report



@dataclass
class TrainParams:
    userdir: str
    pair: str
    timeframe: str = "1h"
    window: int = 128
    total_timesteps: int = 200_000
    seed: int = 42
    model_out_path: str = "models/rl_ppo.zip"
    fee_bps: float = 6.0
    slippage_bps: float = 2.0
    reward_scale: float = 1.0
    pnl_on_close: bool = False
    idle_penalty_bps: float = 0.02
    arch: str = "mlp"  # mlp | lstm | transformer | transformer_big | transformer_hybrid | multiscale
    device: str = "cuda"
    prefer_exchange: Optional[str] = None
    # Env shaping and risk controls
    reward_type: str = "raw"  # raw | vol_scaled | sharpe_proxy
    vol_lookback: int = 20
    turnover_penalty_bps: float = 0.0
    dd_penalty: float = 0.0
    min_hold_bars: int = 0
    cooldown_bars: int = 0
    random_reset: bool = False
    episode_max_steps: int = 0
    # Feature pipeline
    feature_mode: str = "full"  # full | basic
    basic_lookback: int = 64
    extra_timeframes: Optional[List[str]] = None
    # Evaluation
    eval_freq: int = 100_000  # steps; <=0 disables eval
    n_eval_episodes: int = 3
    eval_max_steps: int = 2000
    eval_log_path: Optional[str] = None
    # PPO hyperparams
    ent_coef: float = 0.02
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10



def train_ppo_from_freqtrade_data(params: TrainParams) -> str:
    os.makedirs(os.path.dirname(params.model_out_path), exist_ok=True)

    data_path = _find_data_file(params.userdir, params.pair, params.timeframe, prefer_exchange=params.prefer_exchange)
    if not data_path:
        raise FileNotFoundError(
            f"No parquet found for {params.pair} {params.timeframe} under {params.userdir}/data. "
            f"Run data download first."
        )
    raw = _load_ohlcv(data_path)
    feats = make_features(raw, mode=params.feature_mode, basic_lookback=params.basic_lookback, extra_timeframes=params.extra_timeframes)

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
        idle_penalty_bps=params.idle_penalty_bps,
        reward_type=params.reward_type,
        vol_lookback=params.vol_lookback,
        turnover_penalty_bps=params.turnover_penalty_bps,
        dd_penalty=params.dd_penalty,
        min_hold_bars=params.min_hold_bars,
        cooldown_bars=params.cooldown_bars,
        random_reset=params.random_reset,
        episode_max_steps=params.episode_max_steps,
    )

    def make_train():
        return FTTradingEnv(train_df, tcfg)

    def make_eval():
        env = FTTradingEnv(eval_df, tcfg)
        return Monitor(env)

    env = DummyVecEnv([make_train])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = DummyVecEnv([make_eval])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Compute feature groups for multiscale extractor
    base_cols = list(train_df.columns)
    group_to_indices: Dict[str, List[int]] = {"base": []}
    if params.extra_timeframes:
        for tf in params.extra_timeframes:
            group_to_indices[tf] = []
    for i, col in enumerate(base_cols):
        matched = False
        if params.extra_timeframes:
            for tf in params.extra_timeframes:
                prefix = f"{str(tf).upper()}_"
                if col.startswith(prefix):
                    group_to_indices[tf].append(i)
                    matched = True
                    break
        if not matched:
            group_to_indices["base"].append(i)
    # Add position feature (last column) to base group
    pos_index = len(base_cols)
    group_to_indices["base"].append(pos_index)

    strides: Dict[str, int] = {"base": 1}
    if params.extra_timeframes:
        for tf in params.extra_timeframes:
            strides[tf] = _tf_stride(tf)

    arch = params.arch.lower()
    if arch == "lstm":
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=max(128, params.n_steps // 2),
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(lstm_hidden_size=128, net_arch=[128]),
        )
    elif arch == "transformer":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=TransformerExtractor,
                features_extractor_kwargs=dict(window=params.window, d_model=96, nhead=4, num_layers=2, ff_dim=192),
            ),
        )
    elif arch == "transformer_hybrid":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=HybridLSTMTransformerExtractor,
                features_extractor_kwargs=dict(
                    window=params.window,
                    d_model=128,
                    nhead=4,
                    num_layers=2,
                    ff_dim=256,
                    dropout=0.1,
                    lstm_hidden=128,
                    bidirectional=True,
                ),
            ),
        )
    elif arch == "transformer_big":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=max(1, min(params.n_epochs, 20)),
            ent_coef=max(0.0, params.ent_coef),
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=TransformerExtractor,
                features_extractor_kwargs=dict(window=params.window, d_model=192, nhead=8, num_layers=4, ff_dim=768),
            ),
        )
    elif arch == "multiscale":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=MultiScaleHTFExtractor,
                features_extractor_kwargs=dict(
                    window=params.window,
                    groups=group_to_indices,
                    strides=strides,
                    d_model=128,
                    ff_dim=256,
                    nhead=4,
                    num_layers=1,
                    dropout=0.1,
                ),
            ),
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    callbacks = []
    if params.eval_freq and params.eval_freq > 0:
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(params.model_out_path),
            log_path=os.path.dirname(params.model_out_path),
            eval_freq=params.eval_freq,
            n_eval_episodes=int(params.n_eval_episodes),
            deterministic=True,
            render=False,
        ))
        callbacks.append(PnLEvalCallback(eval_env, eval_freq=params.eval_freq, max_steps=params.eval_max_steps, deterministic=True, verbose=1, eval_log_path=params.eval_log_path))

    # Sync eval normalization stats with training env
    if isinstance(env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = env.obs_rms

    cb = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=int(params.total_timesteps), callback=cb, progress_bar=True)
    model.save(params.model_out_path)
    # Persist training feature set for consistent inference
    try:
        import json as _json
        feat_path = os.path.join(os.path.dirname(params.model_out_path), "feature_columns.json")
        with open(feat_path, "w") as f:
            _json.dump(list(train_df.columns), f)
    except Exception:
        pass
    # Save normalization statistics alongside the model
    if isinstance(env, VecNormalize):
        stats_path = os.path.join(os.path.dirname(params.model_out_path), "vecnormalize.pkl")
        env.save(stats_path)
    # Quick validation rollout on eval split
    try:
        _ = _run_validation_rollout(model, eval_env, max_steps=1000, deterministic=True)
    except Exception:
        pass
    return params.model_out_path



def train_ppo_multi_from_freqtrade_data(params: TrainParams, pairs: List[str]) -> str:
    """Train PPO on multiple symbols simultaneously using a vectorized environment.

    All datasets must share the same feature layout. We construct one env per symbol
    (train split for each), and a single eval env on the first symbol's eval split.
    """
    import pandas as _pd  # local import to avoid polluting module namespace

    # Load datasets and compute features for each
    raw_dfs: List[_pd.DataFrame] = []
    feat_train: List[_pd.DataFrame] = []
    feat_eval: List[_pd.DataFrame] = []

    for pr in pairs:
        data_path = _find_data_file(params.userdir, pr, params.timeframe, prefer_exchange=params.prefer_exchange)
        if not data_path:
            raise FileNotFoundError(
                f"No parquet found for {pr} {params.timeframe} under {params.userdir}/data. Run data download first."
            )
        raw = _load_ohlcv(data_path)
        feats = make_features(
            raw,
            mode=params.feature_mode,
            basic_lookback=params.basic_lookback,
            extra_timeframes=params.extra_timeframes,
        )
        raw_dfs.append(raw)
        # Split
        n = len(feats)
        cut = int(n * 0.8)
        feat_train.append(feats.iloc[:cut].copy())
        feat_eval.append(feats.iloc[cut - max(params.window, 1):].copy())

    # Sanity: ensure consistent columns across all datasets
    base_cols = list(feat_train[0].columns)
    for df in feat_train[1:]:
        if list(df.columns) != base_cols:
            missing = [c for c in base_cols if c not in df.columns]
            extra = [c for c in df.columns if c not in base_cols]
            raise ValueError(f"Feature column mismatch across datasets. Missing={missing} extra={extra}")

    tcfg = TradingConfig(
        window=params.window,
        fee_bps=params.fee_bps,
        slippage_bps=params.slippage_bps,
        reward_scale=params.reward_scale,
        pnl_on_close=params.pnl_on_close,
        idle_penalty_bps=params.idle_penalty_bps,
        reward_type=params.reward_type,
        vol_lookback=params.vol_lookback,
        turnover_penalty_bps=params.turnover_penalty_bps,
        dd_penalty=params.dd_penalty,
        min_hold_bars=params.min_hold_bars,
        cooldown_bars=params.cooldown_bars,
        random_reset=params.random_reset,
        episode_max_steps=params.episode_max_steps,
    )

    def make_train_env(df: _pd.DataFrame):
        return lambda: FTTradingEnv(df, tcfg)

    def make_eval_env(df: _pd.DataFrame):
        env = FTTradingEnv(df, tcfg)
        return Monitor(env)

    # Vectorized training env across symbols
    train_env = DummyVecEnv([make_train_env(df) for df in feat_train])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Single-symbol eval env (first symbol)
    eval_env = DummyVecEnv([lambda: make_eval_env(feat_eval[0])])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Feature groups for multiscale extractor (based on first dataset)
    group_to_indices: Dict[str, List[int]] = {"base": []}
    if params.extra_timeframes:
        for tf in params.extra_timeframes:
            group_to_indices[tf] = []
    for i, col in enumerate(base_cols):
        matched = False
        if params.extra_timeframes:
            for tf in params.extra_timeframes:
                prefix = f"{str(tf).upper()}_"
                if col.startswith(prefix):
                    group_to_indices[tf].append(i)
                    matched = True
                    break
        if not matched:
            group_to_indices["base"].append(i)
    pos_index = len(base_cols)
    group_to_indices["base"].append(pos_index)

    strides: Dict[str, int] = {"base": 1}
    if params.extra_timeframes:
        for tf in params.extra_timeframes:
            strides[tf] = _tf_stride(tf)

    arch = params.arch.lower()
    if arch == "lstm":
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=max(128, params.n_steps // 2),
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(lstm_hidden_size=128, net_arch=[128]),
        )
    elif arch == "transformer":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=TransformerExtractor,
                features_extractor_kwargs=dict(window=params.window, d_model=96, nhead=4, num_layers=2, ff_dim=192),
            ),
        )
    elif arch == "transformer_hybrid":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=HybridLSTMTransformerExtractor,
                features_extractor_kwargs=dict(
                    window=params.window,
                    d_model=128,
                    nhead=4,
                    num_layers=2,
                    ff_dim=256,
                    dropout=0.1,
                    lstm_hidden=128,
                    bidirectional=True,
                ),
            ),
        )
    elif arch == "transformer_big":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=max(1, min(params.n_epochs, 20)),
            ent_coef=max(0.0, params.ent_coef),
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=TransformerExtractor,
                features_extractor_kwargs=dict(window=params.window, d_model=192, nhead=8, num_layers=4, ff_dim=768),
            ),
        )
    elif arch == "multiscale":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(
                net_arch=[],
                features_extractor_class=MultiScaleHTFExtractor,
                features_extractor_kwargs=dict(
                    window=params.window,
                    groups=group_to_indices,
                    strides=strides,
                    d_model=128,
                    ff_dim=256,
                    nhead=4,
                    num_layers=1,
                    dropout=0.1,
                ),
            ),
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=params.seed,
            device=params.device,
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=params.n_epochs,
            ent_coef=params.ent_coef,
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    callbacks = []
    if params.eval_freq and params.eval_freq > 0:
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(params.model_out_path),
            log_path=os.path.dirname(params.model_out_path),
            eval_freq=params.eval_freq,
            n_eval_episodes=int(params.n_eval_episodes),
            deterministic=True,
            render=False,
        ))
        callbacks.append(PnLEvalCallback(eval_env, eval_freq=params.eval_freq, max_steps=params.eval_max_steps, deterministic=True, verbose=1, eval_log_path=params.eval_log_path))

    # Sync eval normalization stats with training env
    if isinstance(train_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = train_env.obs_rms

    cb = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=int(params.total_timesteps), callback=cb, progress_bar=True)
    model.save(params.model_out_path)

    # Persist training feature set for consistent inference
    try:
        import json as _json
        feat_path = os.path.join(os.path.dirname(params.model_out_path), "feature_columns.json")
        with open(feat_path, "w") as f:
            _json.dump(base_cols, f)
    except Exception:
        pass
    # Save normalization statistics alongside the model
    if isinstance(train_env, VecNormalize):
        stats_path = os.path.join(os.path.dirname(params.model_out_path), "vecnormalize.pkl")
        train_env.save(stats_path)

    # Quick validation rollout on first symbol
    try:
        _ = _run_validation_rollout(model, eval_env, max_steps=1000, deterministic=True)
    except Exception:
        pass
    return params.model_out_path

