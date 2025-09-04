"""
Orchestrator CLI for RL trading pipeline using:
- Freqtrade for data download/backtesting
- Stable-Baselines3 (PPO) for training

Subcommands:
    python rl_trader.py download --pair BTC/USDT --timeframe 1h --timerange 20190101-20240101
    python rl_trader.py train --pair BTC/USDT --timeframe 1h --window 128 --timesteps 200000
    python rl_trader.py backtest --pair BTC/USDT --timeframe 1h --timerange 20220101-20240101
"""

import os
import json
import subprocess
from pathlib import Path
import typer
from itertools import product
from datetime import datetime
import csv
import glob
import zipfile
from typing import Optional, Dict, Any

from rl_lib.train_sb3 import TrainParams, train_ppo_from_freqtrade_data, validate_trained_model
from rl_lib.train_sb3 import train_ppo_multi_from_freqtrade_data  # type: ignore
from rl_lib.train_sb3 import _find_data_file as _find_data_file_internal, _load_ohlcv as _load_ohlcv_internal  # type: ignore
from rl_lib.signal import compute_rl_signals
from rl_lib.forecast_transformer import ForecastTrainParams, train_transformer_forecaster  # type: ignore
from rl_lib.forecast_transformer import evaluate_forecaster  # type: ignore


app = typer.Typer(add_completion=False)


def _parse_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: Any) -> int:
    try:
        if x is None:
            return 0
        return int(x)
    except Exception:
        return 0


@app.command()
def download(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    timerange: str = typer.Option("20190101-20240101", help="YYYYMMDD-YYYYMMDD"),
    exchange: str = typer.Option("binance", help="Exchange name for freqtrade download-data"),
):
    """Download data via Freqtrade to userdir/data."""
    Path(userdir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "freqtrade", "download-data",
        "--pairs", pair,
        "--timeframes", timeframe,
        "--userdir", userdir,
        "--timerange", timerange,
        "--exchange", exchange,
        "--data-format-ohlcv", "parquet",
    ]
    typer.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _ensure_dataset(userdir: str, pair: str, timeframe: str, exchange: str, timerange: str = "20190101-", fmt: str = "parquet"):
    """Ensure dataset exists; if not, download via Freqtrade non-interactively.

    Tries a few pair naming variants and, for Bybit futures, adds trading-mode flags.
    """
    from rl_lib.train_sb3 import _find_data_file as _find
    # Fast path
    hit = _find(userdir, pair, timeframe, prefer_exchange=exchange)
    if hit and os.path.exists(hit):
        return hit

    Path(userdir).mkdir(parents=True, exist_ok=True)

    # Pair variants
    variants = {pair}
    up = pair.upper()
    if ":" not in pair and (up.endswith("/USDT") or up.endswith("_USDT")):
        variants.add(f"{pair}:USDT")

    exc_l = str(exchange).lower()
    last_err: Optional[Exception] = None
    for pv in variants:
        # Base download cmd (no margin/trading-mode flags; not supported by download-data)
        cmd = [
            "freqtrade", "download-data",
            "--pairs", pv,
            "--timeframes", timeframe,
            "--userdir", userdir,
            "--timerange", timerange,
            "--exchange", exchange,
            "--data-format-ohlcv", fmt,
        ]
        try:
            typer.echo(f"Downloading missing dataset: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            last_err = e
            continue
        # Check after attempt
        hit = _find(userdir, pv, timeframe, prefer_exchange=exchange)
        if hit and os.path.exists(hit):
            return hit

    # Final attempt: return whatever was initially found (None) or raise
    if last_err is not None:
        typer.echo(f"Download attempts failed for variants {list(variants)}: {last_err}")
    return _find(userdir, pair, timeframe, prefer_exchange=exchange)


def _ensure_backtest_config(userdir: str, timeframe: str, exchange: str, pair: str) -> Path:
    """Ensure minimal freqtrade config.json exists for backtesting and return its path."""
    config_path = Path(userdir) / "config.json"
    if not config_path.exists():
        cfg = {
            "timeframe": timeframe,
            "user_data_dir": str(userdir),
            "strategy": "RLStrategy",
            "exchange": {
                "name": exchange,
                "key": "",
                "secret": "",
                "pair_whitelist": [pair]
            },
            "stake_currency": "USDT",
            "stake_amount": "unlimited",
            "dry_run": True,
            "max_open_trades": 1,
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "dataformat_ohlcv": "parquet"
        }
        os.makedirs(userdir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return config_path


def _find_latest_backtest_artifacts(backtest_dir: Path) -> Dict[str, Optional[str]]:
    """Return paths to the latest backtest zip and meta.json if present."""
    zips = sorted(glob.glob(str(backtest_dir / "backtest-result-*.zip")))
    metas = sorted(glob.glob(str(backtest_dir / "backtest-result-*.meta.json")))
    return {
        "zip": zips[-1] if zips else None,
        "meta": metas[-1] if metas else None,
    }


def _extract_metrics_from_backtest_zip(zip_path: str) -> Dict[str, Any]:
    """Best-effort parse of metrics from Freqtrade backtest result zip.

    Tries to locate a JSON file inside and extract common metrics if available.
    Returns empty dict on failure.
    """
    out: Dict[str, Any] = {}

    def _find_first_numeric(d: Any, keys: list[str]) -> Optional[float]:
        # Recursive search by key name across nested dict/list
        try:
            if isinstance(d, dict):
                # Strategy-nested single key unwrap (e.g., {"RLStrategy": {...}})
                if len(d) == 1:
                    only_key = next(iter(d))
                    inner = d[only_key]
                    if isinstance(inner, (dict, list)):
                        val = _find_first_numeric(inner, keys)
                        if val is not None:
                            return val
                for k, v in d.items():
                    if isinstance(k, str) and k in keys and isinstance(v, (int, float)):
                        return float(v)
                for v in d.values():
                    val = _find_first_numeric(v, keys)
                    if val is not None:
                        return val
            elif isinstance(d, list):
                for it in d:
                    val = _find_first_numeric(it, keys)
                    if val is not None:
                        return val
        except Exception:
            return None
        return None
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Prefer files that end with .json and contain 'result' or 'results'
            json_names = [n for n in zf.namelist() if n.lower().endswith('.json')]
            # Heuristic ordering
            json_names.sort(key=lambda n: ("result" not in n.lower(), n))
            for name in json_names:
                try:
                    with zf.open(name) as fh:
                        data = json.load(fh)
                        # Try aggregate keys (possibly nested)
                        agg_keys = {
                            'bt_total_profit_abs': ['profit_total_abs', 'total_profit', 'net_profit'],
                            'bt_total_profit_pct': ['profit_total_pct', 'total_profit_pct'],
                            'bt_total_trades': ['total_trades', 'trades'],
                            'bt_win_rate': ['winrate', 'win_rate'],
                            'bt_profit_factor': ['profit_factor', 'pf']
                        }
                        got_any = False
                        for out_k, keys in agg_keys.items():
                            val = _find_first_numeric(data, keys)
                            if val is not None:
                                out[out_k] = val
                                got_any = True

                        # If aggregates missing, compute from trades list if present
                        if not got_any:
                            trades_list = None
                            # Common names inside exports
                            for tkey in ['trades', 'closed_trades', 'results_trades', 'backtest_trades']:
                                if isinstance(data, dict) and tkey in data and isinstance(data[tkey], list):
                                    trades_list = data[tkey]
                                    break
                            if trades_list is None and isinstance(data, list):
                                # Some exports are a raw trades array
                                trades_list = data
                            if isinstance(trades_list, list) and trades_list:
                                total_trades = len(trades_list)
                                wins = 0
                                pl_sum = 0.0
                                pos_sum = 0.0
                                neg_sum = 0.0
                                for tr in trades_list:
                                    try:
                                        p = None
                                        # Prefer absolute profit if available
                                        if isinstance(tr, dict):
                                            if 'profit_abs' in tr and isinstance(tr['profit_abs'], (int, float)):
                                                p = float(tr['profit_abs'])
                                            elif 'profit' in tr and isinstance(tr['profit'], (int, float)):
                                                p = float(tr['profit'])
                                            elif 'profit_ratio' in tr and isinstance(tr['profit_ratio'], (int, float)):
                                                p = float(tr['profit_ratio'])
                                        if p is not None:
                                            pl_sum += p
                                            if p > 0:
                                                wins += 1
                                                pos_sum += p
                                            elif p < 0:
                                                neg_sum += p
                                    except Exception:
                                        continue
                                out['bt_total_trades'] = float(total_trades)
                                if total_trades > 0:
                                    out['bt_win_rate'] = 100.0 * (wins / total_trades)
                                # Use sums as proxies
                                out['bt_total_profit_abs'] = pl_sum
                                if pos_sum > 0 and neg_sum < 0:
                                    out['bt_profit_factor'] = pos_sum / abs(neg_sum)

                        # Stop at first JSON that yields any metric
                        if any(k in out for k in ('bt_total_profit_abs','bt_total_profit_pct','bt_total_trades','bt_win_rate','bt_profit_factor')):
                            out['bt_json_in_zip'] = name
                            break
                except Exception:
                    continue
    except Exception:
        return {}
    return out


def _run_freqtrade_backtest(*, userdir: str, pair: str, timeframe: str, exchange: str, timerange: str, model_path: str, window: int, device: str, min_hold_bars: int, cooldown_bars: int) -> Dict[str, Any]:
    """Run freqtrade backtesting for the given model and return artifact/metrics info."""
    env = os.environ.copy()
    env["RL_DEVICE"] = device
    env["RL_MODEL_PATH"] = model_path
    env["RL_WINDOW"] = str(window)
    env["RL_MIN_HOLD_BARS"] = str(int(min_hold_bars))
    env["RL_COOLDOWN_BARS"] = str(int(cooldown_bars))

    config_path = _ensure_backtest_config(userdir, timeframe, exchange, pair)
    backtest_dir = Path(userdir) / "backtest_results"
    pre_existing = set(os.listdir(backtest_dir)) if backtest_dir.exists() else set()

    cmd = [
        "freqtrade", "backtesting",
        "--userdir", userdir,
        "--config", str(config_path),
        "--strategy", "RLStrategy",
        "--timeframe", timeframe,
        "--pairs", pair,
    ]
    if timerange:
        cmd.extend(["--timerange", timerange])
    # Ensure trade exports are included in zip for metric parsing
    cmd.extend(["--export", "trades"])
    try:
        subprocess.run(cmd, check=True, env=env)
    except Exception as e:
        return {"bt_error": str(e), "bt_cmd": " ".join(cmd)}

    # Detect newly created artifacts
    backtest_dir.mkdir(parents=True, exist_ok=True)
    post_existing = set(os.listdir(backtest_dir))
    new_files = post_existing - pre_existing
    artifacts = _find_latest_backtest_artifacts(backtest_dir)

    result: Dict[str, Any] = {
        "bt_zip": artifacts.get("zip"),
        "bt_meta": artifacts.get("meta"),
        "bt_cmd": " ".join(cmd),
    }
    # Attach meta info if present
    try:
        if result.get("bt_meta") and os.path.exists(result["bt_meta"]):
            with open(result["bt_meta"], "r") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                # Expect strategy name key like 'RLStrategy'
                for k, v in meta.items():
                    if isinstance(v, dict):
                        result['bt_run_id'] = v.get('run_id')
                        result['bt_timeframe'] = v.get('timeframe')
                        result['bt_start_ts'] = v.get('backtest_start_ts')
                        result['bt_end_ts'] = v.get('backtest_end_ts')
                        break
    except Exception:
        pass
    # Try to parse metrics from zip
    try:
        if result.get("bt_zip") and os.path.exists(result["bt_zip"]):
            result.update(_extract_metrics_from_backtest_zip(result["bt_zip"]))
    except Exception:
        pass
    result['bt_new_files_count'] = len(new_files)
    return result


@app.command()
def train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    window: int = typer.Option(128),
    timesteps: int = typer.Option(200_000),
    model_out: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "rl_ppo.zip"), "--model-out", "--model_out"),
    arch: str = typer.Option("mlp", help="mlp|lstm|transformer|transformer_big|transformer_hybrid|multiscale"),
    fee_bps: float = typer.Option(6.0, help="Trading fee in basis points (e.g., 6.0 = 0.06%)"),
    slippage_bps: float = typer.Option(2.0, help="Slippage in basis points"),
    idle_penalty_bps: float = typer.Option(0.02, help="Idle penalty in bps when flat (applied in env)"),
    exchange: str = typer.Option("bybit", help="Prefer this exchange's dataset when multiple exist"),
    device: str = typer.Option("cuda", help="Device for training: cuda|cpu"),
    # Auto-download
    autofetch: bool = typer.Option(True, help="Auto-download missing datasets (1h,4h,1d,1w)"),
    timerange: str = typer.Option("20190101-", help="Timerange for auto-download"),
    # Training data slicing
    train_timerange: str = typer.Option("", help="Optional timerange YYYYMMDD-YYYYMMDD to slice training dataset"),
    # New options
    seed: int = typer.Option(42, help="Random seed"),
    reward_type: str = typer.Option("raw", help="raw|vol_scaled|sharpe_proxy"),
    vol_lookback: int = typer.Option(20, help="Volatility lookback for reward shaping"),
    turnover_penalty_bps: float = typer.Option(0.0, help="Extra turnover penalty in bps per change"),
    dd_penalty: float = typer.Option(0.0, help="Coefficient for drawdown penalty in sharpe_proxy"),
    # Risk shaping (exposure control)
    position_penalty_bps: float = typer.Option(0.0, help="Per-step penalty in bps when in position"),
    loss_hold_penalty_bps: float = typer.Option(0.0, help="Per-step penalty in bps times consecutive loss bars while holding"),
    cvar_alpha: float = typer.Option(0.0, help="Tail quantile for CVaR penalty, e.g., 0.05"),
    cvar_coef: float = typer.Option(0.0, help="Scale for CVaR penalty"),
    max_position_bars: int = typer.Option(0, help="Soft cap on bars in position; >0 adds extra penalty beyond this"),
    min_hold_bars: int = typer.Option(0, "--min-hold-bars", "--min_hold_bars", help="Minimum bars to hold before closing/flip"),
    cooldown_bars: int = typer.Option(0, "--cooldown-bars", "--cooldown_bars", help="Cooldown bars after closing before re-entry"),
    random_reset: bool = typer.Option(False, help="Randomize episode start index"),
    episode_max_steps: int = typer.Option(0, help="Max steps per episode (0 = run to dataset end)"),
    feature_mode: str = typer.Option("full", help="full|basic (basic: close_z, change, d_hl)"),
    basic_lookback: int = typer.Option(64, help="Lookback for basic close_z standardization"),
    extra_timeframes: str = typer.Option("", "--extra-timeframes", "--extra_timeframes", help="Optional comma-separated HTFs to include, e.g., '4H,1D'"),
    # Eval options
    eval_freq: int = typer.Option(100000, help="Evaluate every N steps (0 disables)"),
    n_eval_episodes: int = typer.Option(3, help="Episodes per eval"),
    eval_max_steps: int = typer.Option(2000, help="Max steps per eval rollout"),
    eval_log_path: str = typer.Option("", help="CSV path to append eval results"),
    # Early stopping (optional)
    early_stop_metric: str = typer.Option("", help="Metric to monitor for early stop: sharpe|final_equity|max_drawdown"),
    early_stop_patience: int = typer.Option(0, help="Evals to wait without improvement before stopping"),
    early_stop_min_delta: float = typer.Option(0.0, help="Minimum improvement to reset patience"),
    early_stop_degrade_ratio: float = typer.Option(0.0, help="Stop if metric drops below best*(1-ratio)"),
    # PPO overrides
    ent_coef: float = typer.Option(0.02, help="Entropy coefficient"),
    learning_rate: float = typer.Option(3e-4, help="Learning rate"),
    n_steps: int = typer.Option(2048, help="Rollout length per update"),
    batch_size: int = typer.Option(256, help="Batch size"),
    n_epochs: int = typer.Option(10, help="Epochs per update"),
):
    """Train PPO on downloaded data using Stable-Baselines3."""
    etf_list = _parse_list(extra_timeframes, str) if extra_timeframes else []
    if autofetch:
        # Ensure base timeframe and common HTFs
        tfs = sorted(set([timeframe, "1h", "4h", "1d", "1w"]))
        for tf in tfs:
            try:
                _ = _ensure_dataset(userdir, pair, tf, exchange=exchange, timerange=timerange)
            except Exception as e:
                typer.echo(f"Auto-download failed for {pair} {tf}: {e}")
    params = TrainParams(
        userdir=userdir,
        pair=pair,
        timeframe=timeframe,
        window=window,
        total_timesteps=timesteps,
        model_out_path=model_out,
        arch=arch,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        idle_penalty_bps=idle_penalty_bps,
        prefer_exchange=exchange,
        device=device,
        seed=seed,
        reward_type=reward_type,
        vol_lookback=vol_lookback,
        turnover_penalty_bps=turnover_penalty_bps,
        dd_penalty=dd_penalty,
        position_penalty_bps=position_penalty_bps,
        loss_hold_penalty_bps=loss_hold_penalty_bps,
        cvar_alpha=cvar_alpha,
        cvar_coef=cvar_coef,
        max_position_bars=max_position_bars,
        min_hold_bars=min_hold_bars,
        cooldown_bars=cooldown_bars,
        random_reset=random_reset,
        episode_max_steps=episode_max_steps,
        feature_mode=feature_mode,
        basic_lookback=basic_lookback,
        extra_timeframes=etf_list or None,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        eval_max_steps=eval_max_steps,
        eval_log_path=(eval_log_path or None),
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        timerange=(train_timerange or None),
        early_stop_metric=(early_stop_metric or None),
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_degrade_ratio=early_stop_degrade_ratio,
    )
    out = train_ppo_from_freqtrade_data(params)
    typer.echo(f"Model saved: {out}")


@app.command()
def train_multi(
    pairs: str = typer.Option("BTC/USDT:USDT,ETH/USDT:USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    window: int = typer.Option(128),
    timesteps: int = typer.Option(200_000),
    model_out: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "rl_ppo.zip"), "--model-out", "--model_out"),
    arch: str = typer.Option("transformer_hybrid"),
    exchange: str = typer.Option("bybit"),
    device: str = typer.Option("cuda"),
    # Env/training knobs
    seed: int = typer.Option(42),
    reward_type: str = typer.Option("vol_scaled"),
    vol_lookback: int = typer.Option(20),
    fee_bps: float = typer.Option(6.0),
    slippage_bps: float = typer.Option(10.0),
    turnover_penalty_bps: float = typer.Option(2.0),
    min_hold_bars: int = typer.Option(3),
    cooldown_bars: int = typer.Option(1),
    # Risk shaping (exposure control)
    position_penalty_bps: float = typer.Option(0.0),
    loss_hold_penalty_bps: float = typer.Option(0.0),
    cvar_alpha: float = typer.Option(0.0),
    cvar_coef: float = typer.Option(0.0),
    max_position_bars: int = typer.Option(0),
    random_reset: bool = typer.Option(True),
    episode_max_steps: int = typer.Option(4096),
    feature_mode: str = typer.Option("basic"),
    basic_lookback: int = typer.Option(128),
    extra_timeframes: str = typer.Option("4H,1D", "--extra-timeframes", "--extra_timeframes"),
    # Eval knobs
    eval_freq: int = typer.Option(50000, help="Evaluate every N steps (0 disables)"),
    n_eval_episodes: int = typer.Option(1, help="Episodes per eval"),
    eval_max_steps: int = typer.Option(2000, help="Max steps per eval rollout"),
    # Early stopping (optional)
    early_stop_metric: str = typer.Option("", help="Metric to monitor for early stop: sharpe|final_equity|max_drawdown"),
    early_stop_patience: int = typer.Option(0, help="Evals to wait without improvement before stopping"),
    early_stop_min_delta: float = typer.Option(0.0, help="Minimum improvement to reset patience"),
    early_stop_degrade_ratio: float = typer.Option(0.0, help="Stop if metric drops below best*(1-ratio)"),
    # Auto-download knobs
    autofetch: bool = typer.Option(True, help="Auto-download missing datasets (1h,4h,1d,1w)"),
    timerange: str = typer.Option("20190101-", help="Timerange for auto-download"),
    # Training data slicing
    train_timerange: str = typer.Option("", help="Optional timerange YYYYMMDD-YYYYMMDD to slice training datasets"),
    align_mode: str = typer.Option("union", help="multi-train alignment: union|intersection"),
):
    """Train PPO on multiple symbols with vectorized envs. Auto-downloads datasets if missing."""
    pair_list = _parse_list(pairs, str)
    etf_list = _parse_list(extra_timeframes, str) if extra_timeframes else []

    if autofetch:
        # Ensure base timeframe and common HTFs for each pair
        tfs = sorted(set([timeframe, "1h", "4h", "1d", "1w"]))
        for pr in pair_list:
            for tf in tfs:
                try:
                    _ = _ensure_dataset(userdir, pr, tf, exchange=exchange, timerange=timerange)
                except Exception as e:
                    typer.echo(f"Auto-download failed for {pr} {tf}: {e}")

    params = TrainParams(
        userdir=userdir,
        pair=pair_list[0],  # used for metadata paths
        timeframe=timeframe,
        window=window,
        total_timesteps=timesteps,
        model_out_path=model_out,
        arch=arch,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        idle_penalty_bps=0.0,
        prefer_exchange=exchange,
        device=device,
        seed=seed,
        reward_type=reward_type,
        vol_lookback=vol_lookback,
        turnover_penalty_bps=turnover_penalty_bps,
        dd_penalty=0.05 if reward_type == "sharpe_proxy" else 0.0,
        position_penalty_bps=position_penalty_bps,
        loss_hold_penalty_bps=loss_hold_penalty_bps,
        cvar_alpha=cvar_alpha,
        cvar_coef=cvar_coef,
        max_position_bars=max_position_bars,
        min_hold_bars=min_hold_bars,
        cooldown_bars=cooldown_bars,
        random_reset=random_reset,
        episode_max_steps=episode_max_steps,
        feature_mode=feature_mode,
        basic_lookback=basic_lookback,
        extra_timeframes=etf_list or None,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        eval_max_steps=eval_max_steps,
        eval_log_path=str(Path(model_out).with_suffix("") ) + "_eval.csv",
        ent_coef=0.02,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        timerange=(train_timerange or None),
        align_mode=str(align_mode).lower(),
        early_stop_metric=(early_stop_metric or None),
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_degrade_ratio=early_stop_degrade_ratio,
    )
    out = train_ppo_multi_from_freqtrade_data(params, pair_list)
    typer.echo(f"Model saved: {out}")


@app.command()
def backtest(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    model_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "rl_ppo.zip")),
    timerange: str = typer.Option("20250101-"),
    device: str = typer.Option("cuda", help="Device for backtest: cuda|cpu"),
    exchange: str = typer.Option("bybit", help="Exchange name for dataset and backtest config"),
    window: int = typer.Option(128, help="Window size (must match training)"),
):
    """Backtest trained RL model inside Freqtrade."""
    env = os.environ.copy()
    env["RL_DEVICE"] = device
    env["RL_MODEL_PATH"] = model_path
    env["RL_WINDOW"] = str(window)
    # Ensure minimal config exists
    config_path = Path(userdir) / "config.json"
    if not config_path.exists():
        cfg = {
            "timeframe": timeframe,
            "user_data_dir": str(userdir),
            "strategy": "RLStrategy",
            "exchange": {
                "name": exchange,
                "key": "",
                "secret": "",
                "pair_whitelist": [pair]
            },
            "stake_currency": "USDT",
            "stake_amount": "unlimited",
            "dry_run": True,
            "max_open_trades": 1,
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "dataformat_ohlcv": "parquet"
        }
        os.makedirs(userdir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
    cmd = [
        "freqtrade", "backtesting",
        "--userdir", userdir,
        "--config", str(config_path),
        "--strategy", "RLStrategy",
        "--timeframe", timeframe,
        "--pairs", pair,
        "--timerange", timerange,
    ]
    typer.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


@app.command()
def sweep(
    pair: str = typer.Option("BTC/USDT:USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    window: int = typer.Option(128),
    timesteps: int = typer.Option(300_000, help="Timesteps per trial"),
    arch: str = typer.Option("transformer_hybrid"),
    device: str = typer.Option("cuda"),
    # Grid lists (comma-separated)
    reward_types: str = typer.Option("vol_scaled,sharpe_proxy,raw"),
    ent_coefs: str = typer.Option("0.02,0.05,0.1"),
    lrs: str = typer.Option("0.0003,0.0005"),
    n_steps_list: str = typer.Option("1024,2048"),
    batch_sizes: str = typer.Option("256,512"),
    fee_list: str = typer.Option("2,6"),
    slip_list: str = typer.Option("5,10"),
    seeds: str = typer.Option("42,1337"),
    extra_timeframes: str = typer.Option("", help="Optional comma-separated HTFs to include for all trials"),
    # Fixed env knobs
    turnover_penalty_bps: float = typer.Option(1.0),
    min_hold_bars: int = typer.Option(2),
    cooldown_bars: int = typer.Option(0),
    # Risk shaping (exposure control)
    position_penalty_bps: float = typer.Option(0.0),
    loss_hold_penalty_bps: float = typer.Option(0.0),
    cvar_alpha: float = typer.Option(0.0),
    cvar_coef: float = typer.Option(0.0),
    max_position_bars: int = typer.Option(0),
    # Risk shaping list variants for grid search
    position_penalty_bps_list: str = typer.Option("", help="Comma-separated list for position penalty bps"),
    loss_hold_penalty_bps_list: str = typer.Option("", help="Comma-separated list for loss-hold penalty bps"),
    cvar_alpha_list: str = typer.Option("", help="Comma-separated list for CVaR alpha"),
    cvar_coef_list: str = typer.Option("", help="Comma-separated list for CVaR coef"),
    max_position_bars_list: str = typer.Option("", help="Comma-separated list for max position bars"),
    feature_mode: str = typer.Option("basic"),
    basic_lookback: int = typer.Option(128),
    windows_list: str = typer.Option("", help="Comma-separated window sizes; if empty, uses --window"),
    min_hold_bars_list: str = typer.Option("", help="Comma-separated min-hold bars values"),
    cooldown_bars_list: str = typer.Option("", help="Comma-separated cooldown bars values"),
    feature_modes: str = typer.Option("", help="Comma-separated feature modes to sweep: basic,full"),
    # Eval and early stop
    eval_max_steps: int = typer.Option(5000),
    eval_freq: int = typer.Option(50000, help="Evaluate every N steps; enable >0 for early stopping"),
    early_stop_metric: str = typer.Option("sharpe", help="sharpe|final_equity|max_drawdown"),
    early_stop_patience: int = typer.Option(3),
    early_stop_min_delta: float = typer.Option(0.0),
    early_stop_degrade_ratio: float = typer.Option(0.0),
    # Auto backtest
    auto_backtest: bool = typer.Option(True, help="Run freqtrade backtest per model"),
    backtest_timerange: str = typer.Option("20250101-", help="YYYYMMDD-YYYYMMDD for freqtrade backtests; default recent to speed up"),
    backtest_exchange: str = typer.Option("bybit"),
    # Validation slice after training
    eval_timerange: str = typer.Option("20240101-20250101", help="Timerange for post-train validation report"),
    # Auto-download datasets
    autofetch: bool = typer.Option(True, help="Auto-download missing datasets (1h,4h,1d,1w)"),
    timerange: str = typer.Option("20190101-", help="Timerange for auto-download"),
    exchange: str = typer.Option("bybit", help="Exchange name for dataset download and preference"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "sweeps")),
    max_trials: int = typer.Option(50, help="Hard cap to avoid huge grids"),
):
    """Run a small hyperparameter grid sequentially and log results to CSV."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(outdir) / ts
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_csv = sweep_dir / "results.csv"
    fields = [
        "model_path","feature_mode","window","min_hold_bars","cooldown_bars",
        "position_penalty_bps","loss_hold_penalty_bps","cvar_alpha","cvar_coef","max_position_bars",
        "turnover_penalty_bps","extra_timeframes","reward_type","ent_coef","learning_rate","n_steps","batch_size","fee_bps","slippage_bps","seed",
        "eval_timerange","backtest_timerange","exchange",
        "final_equity","sharpe","max_drawdown","time_in_position_frac",
        "bt_total_profit_pct","bt_total_profit_abs","bt_total_trades","bt_win_rate","bt_profit_factor",
        "bt_zip","bt_meta","bt_run_id"
    ]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    rt_list = _parse_list(reward_types, str)
    ec_list = _parse_list(ent_coefs, float)
    lr_list = _parse_list(lrs, float)
    ns_list = _parse_list(n_steps_list, int)
    bs_list = _parse_list(batch_sizes, int)
    fee_vals = _parse_list(fee_list, float)
    slip_vals = _parse_list(slip_list, float)
    seed_vals = _parse_list(seeds, int)
    etf_list = _parse_list(extra_timeframes, str) if extra_timeframes else []
    win_vals = _parse_list(windows_list, int) if windows_list else [window]
    mh_vals = _parse_list(min_hold_bars_list, int) if min_hold_bars_list else [min_hold_bars]
    cd_vals = _parse_list(cooldown_bars_list, int) if cooldown_bars_list else [cooldown_bars]
    fm_list = [m.strip() for m in feature_modes.split(",") if m.strip()] if feature_modes else [feature_mode]
    # Risk shaping grids (fallback to scalar if list not provided)
    pp_vals = _parse_list(position_penalty_bps_list, float) if position_penalty_bps_list else [position_penalty_bps]
    lh_vals = _parse_list(loss_hold_penalty_bps_list, float) if loss_hold_penalty_bps_list else [loss_hold_penalty_bps]
    ca_vals = _parse_list(cvar_alpha_list, float) if cvar_alpha_list else [cvar_alpha]
    cc_vals = _parse_list(cvar_coef_list, float) if cvar_coef_list else [cvar_coef]
    mpb_vals = _parse_list(max_position_bars_list, int) if max_position_bars_list else [max_position_bars]

    # Ensure datasets exist if requested
    if autofetch:
        tfs = sorted(set([timeframe, "1h", "4h", "1d", "1w"]))
        for tf in tfs:
            try:
                _ = _ensure_dataset(userdir, pair, tf, exchange=exchange, timerange=timerange)
            except Exception as e:
                typer.echo(f"Auto-download failed for {pair} {tf}: {e}")

    combos = list(product(
        rt_list, ec_list, lr_list, ns_list, bs_list, fee_vals, slip_vals, seed_vals,
        win_vals, mh_vals, cd_vals, fm_list, pp_vals, lh_vals, ca_vals, cc_vals, mpb_vals
    ))
    if len(combos) > max_trials:
        combos = combos[:max_trials]

    typer.echo(f"Running {len(combos)} trials. Results -> {results_csv}")

    for idx, (rt, ec, lr, ns, bs, fee, slip, sd, wv, mhv, cdv, fm, ppv, lhv, cav, ccv, mpbv) in enumerate(combos, start=1):
        tag = f"{arch}_fm-{fm}_win-{wv}_mh-{mhv}_cd-{cdv}_rt-{rt}_ec-{ec}_lr-{lr}_ns-{ns}_bs-{bs}_fee-{fee}_slip-{slip}_seed-{sd}"
        model_path = str(sweep_dir / f"{tag}.zip")
        params = TrainParams(
            userdir=userdir,
            pair=pair,
            timeframe=timeframe,
            window=wv,
            total_timesteps=timesteps,
            model_out_path=model_path,
            arch=arch,
            device=device,
            prefer_exchange=exchange,
            fee_bps=fee,
            slippage_bps=slip,
            idle_penalty_bps=0.0,
            reward_type=rt,
            vol_lookback=20,
            turnover_penalty_bps=turnover_penalty_bps,
            dd_penalty=0.05 if rt == "sharpe_proxy" else 0.0,
            position_penalty_bps=ppv,
            loss_hold_penalty_bps=lhv,
            cvar_alpha=cav,
            cvar_coef=ccv,
            max_position_bars=mpbv,
            min_hold_bars=mhv,
            cooldown_bars=cdv,
            random_reset=True,
            episode_max_steps=4096,
            feature_mode=fm,
            basic_lookback=basic_lookback,
            extra_timeframes=etf_list or None,
            eval_freq=eval_freq,
            n_eval_episodes=1,
            eval_max_steps=eval_max_steps,
            eval_log_path=str(sweep_dir / "eval_log.csv"),
            seed=sd,
            ent_coef=ec,
            learning_rate=lr,
            n_steps=ns,
            batch_size=bs,
            n_epochs=10,
            early_stop_metric=(early_stop_metric or None),
            early_stop_patience=int(early_stop_patience),
            early_stop_min_delta=float(early_stop_min_delta),
            early_stop_degrade_ratio=float(early_stop_degrade_ratio),
        )
        typer.echo(f"[{idx}/{len(combos)}] Training {tag}")
        try:
            _ = train_ppo_from_freqtrade_data(params)
            report = validate_trained_model(params, max_steps=eval_max_steps, deterministic=True, timerange=eval_timerange)
            bt_metrics: Dict[str, Any] = {}
            if auto_backtest:
                bt_metrics = _run_freqtrade_backtest(
                    userdir=userdir,
                    pair=pair,
                    timeframe=timeframe,
                    exchange=backtest_exchange,
                    timerange=(backtest_timerange or (params.timerange or "")) or "",
                    model_path=model_path,
                    window=wv,
                    device=device,
                    min_hold_bars=mhv,
                    cooldown_bars=cdv,
                )
            row = {
                "model_path": model_path,
                "feature_mode": str(fm),
                "window": int(wv),
                "min_hold_bars": int(mhv),
                "cooldown_bars": int(cdv),
                "position_penalty_bps": _safe_float(ppv),
                "loss_hold_penalty_bps": _safe_float(lhv),
                "cvar_alpha": _safe_float(cav),
                "cvar_coef": _safe_float(ccv),
                "max_position_bars": _safe_int(mpbv),
                "turnover_penalty_bps": _safe_float(turnover_penalty_bps),
                "extra_timeframes": ",".join(etf_list) if etf_list else "",
                "reward_type": rt,
                "ent_coef": _safe_float(ec),
                "learning_rate": _safe_float(lr),
                "n_steps": _safe_int(ns),
                "batch_size": _safe_int(bs),
                "fee_bps": _safe_float(fee),
                "slippage_bps": _safe_float(slip),
                "seed": _safe_int(sd),
                "eval_timerange": str(eval_timerange),
                "backtest_timerange": str(backtest_timerange or ""),
                "exchange": str(exchange),
                "final_equity": _safe_float(report.get("final_equity")),
                "sharpe": _safe_float(report.get("sharpe")),
                "max_drawdown": _safe_float(report.get("max_drawdown")),
                "time_in_position_frac": _safe_float(report.get("time_in_position_frac")),
                "bt_total_profit_pct": _safe_float(bt_metrics.get("bt_total_profit_pct")) if auto_backtest else float("nan"),
                "bt_total_profit_abs": _safe_float(bt_metrics.get("bt_total_profit_abs")) if auto_backtest else float("nan"),
                "bt_total_trades": _safe_int(bt_metrics.get("bt_total_trades")) if auto_backtest else 0,
                "bt_win_rate": _safe_float(bt_metrics.get("bt_win_rate")) if auto_backtest else float("nan"),
                "bt_profit_factor": _safe_float(bt_metrics.get("bt_profit_factor")) if auto_backtest else float("nan"),
                "bt_zip": str(bt_metrics.get("bt_zip", "")) if auto_backtest else "",
                "bt_meta": str(bt_metrics.get("bt_meta", "")) if auto_backtest else "",
                "bt_run_id": str(bt_metrics.get("bt_run_id", "")) if auto_backtest else "",
            }
            with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(row)
        except Exception as e:
            typer.echo(f"Trial failed: {tag} -> {e}")
            continue

    typer.echo("Sweep complete.")


@app.command()
def validate(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    window: int = typer.Option(128),
    model_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "rl_ppo.zip"), "--model-path", "--model_path"),
    max_steps: int = typer.Option(1000),
    deterministic: bool = typer.Option(True),
    device: str = typer.Option("cuda", help="Device for validation: cuda|cpu"),
    exchange: str = typer.Option("bybit", help="Prefer this exchange's dataset when multiple exist"),
    fee_bps: float = typer.Option(6.0, help="Trading fee in basis points (e.g., 6.0 = 0.06%)"),
    slippage_bps: float = typer.Option(2.0, help="Slippage in basis points"),
    idle_penalty_bps: float = typer.Option(0.0, help="Idle penalty in bps when flat (applied in env)"),
    timerange: str = typer.Option("", help="Optional timerange YYYYMMDD-YYYYMMDD for validation dataset"),
    # New options
    reward_type: str = typer.Option("raw", help="raw|vol_scaled|sharpe_proxy"),
    vol_lookback: int = typer.Option(20, help="Volatility lookback for reward shaping"),
    turnover_penalty_bps: float = typer.Option(0.0, help="Extra turnover penalty in bps per change"),
    dd_penalty: float = typer.Option(0.0, help="Coefficient for drawdown penalty in sharpe_proxy"),
    min_hold_bars: int = typer.Option(0, "--min-hold-bars", "--min_hold_bars", help="Minimum bars to hold before closing/flip"),
    cooldown_bars: int = typer.Option(0, "--cooldown-bars", "--cooldown_bars", help="Cooldown bars after closing before re-entry"),
    random_reset: bool = typer.Option(False, help="Randomize episode start index"),
    episode_max_steps: int = typer.Option(0, help="Max steps per episode (0 = run to dataset end)"),
    feature_mode: str = typer.Option("full", help="full|basic (basic: close_z, change, d_hl)"),
    basic_lookback: int = typer.Option(64, help="Lookback for basic close_z standardization"),
    extra_timeframes: str = typer.Option("", "--extra-timeframes", "--extra_timeframes", help="Optional comma-separated HTFs to include, e.g., '4H,1D'"),
):
    """Run a quick validation rollout on eval split and print summary (actions, entries, equity)."""
    etf_list = _parse_list(extra_timeframes, str) if extra_timeframes else []
    params = TrainParams(
        userdir=userdir,
        pair=pair,
        timeframe=timeframe,
        window=window,
        model_out_path=model_path,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        idle_penalty_bps=idle_penalty_bps,
        prefer_exchange=exchange,
        reward_type=reward_type,
        vol_lookback=vol_lookback,
        turnover_penalty_bps=turnover_penalty_bps,
        dd_penalty=dd_penalty,
        min_hold_bars=min_hold_bars,
        cooldown_bars=cooldown_bars,
        random_reset=random_reset,
        episode_max_steps=episode_max_steps,
        feature_mode=feature_mode,
        basic_lookback=basic_lookback,
        extra_timeframes=etf_list or None,
    )
    os.environ["RL_DEVICE"] = device
    _ = validate_trained_model(params, max_steps=max_steps, deterministic=deterministic, timerange=timerange)


@app.command()
def forecast_train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    # Data slice
    timerange: str = typer.Option("", help="YYYYMMDD-YYYYMMDD"),
    feature_mode: str = typer.Option("full", help="full|basic|ohlcv"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("", help="e.g., '4H,1D'"),
    # Model/data
    window: int = typer.Option(128),
    horizon: int = typer.Option(16),
    target_columns: str = typer.Option("open,high,low,close,volume", help="Comma-separated targets; empty=auto OHLCV if present"),
    target_mode: str = typer.Option("price", help="price|logret (predict next-step log return)"),
    forecast_arch: str = typer.Option("encdec", help="encdec|decoder_only"),
    d_model: int = typer.Option(128),
    nhead: int = typer.Option(4),
    num_encoder_layers: int = typer.Option(2),
    num_decoder_layers: int = typer.Option(2),
    ff_dim: int = typer.Option(256),
    dropout: float = typer.Option(0.1),
    # Training
    batch_size: int = typer.Option(128),
    epochs: int = typer.Option(10),
    learning_rate: float = typer.Option(3e-4),
    weight_decay: float = typer.Option(1e-4),
    grad_clip_norm: float = typer.Option(1.0),
    device: str = typer.Option("cuda", help="cuda|cpu"),
    seed: int = typer.Option(42),
    # Early stop / LR schedule
    early_stop_patience: int = typer.Option(0, help="Stop if no val improvement for N epochs (0=off)"),
    lr_plateau_patience: int = typer.Option(0, help="Reduce LR on plateau after N epochs (0=off)"),
    lr_factor: float = typer.Option(0.5, help="LR reduction factor on plateau"),
    min_lr: float = typer.Option(1e-6, help="Minimum learning rate for scheduler"),
    # Output
    model_out: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "forecaster.pt"), "--model-out", "--model_out"),
    # Auto-download
    autofetch: bool = typer.Option(True, help="Auto-download 1h,4h,1d,1w datasets if missing"),
    download_timerange: str = typer.Option("20190101-", help="Timerange for auto-download (YYYYMMDD-YYYYMMDD)"),
    exchange: str = typer.Option("bybit", help="Prefer this exchange's dataset when multiple exist"),
):
    """Train a Transformer decoder forecaster to predict next N candles (autoregressive)."""
    etf_list = [s.strip() for s in extra_timeframes.split(",") if s.strip()] if extra_timeframes else []
    tcols = [s.strip() for s in target_columns.split(",") if s.strip()] if target_columns else None
    # Optional auto-download
    if autofetch:
        tfs = sorted(set([timeframe, "1h", "4h", "1d", "1w"]))
        for tf in tfs:
            try:
                _ = _ensure_dataset(userdir, pair, tf, exchange=exchange, timerange=download_timerange)
            except Exception as e:
                typer.echo(f"Auto-download failed for {pair} {tf}: {e}")

    params = ForecastTrainParams(
        userdir=userdir,
        pair=pair,
        timeframe=timeframe,
        feature_mode=feature_mode,
        basic_lookback=basic_lookback,
        extra_timeframes=(etf_list or None),
        timerange=(timerange or None),
        window=window,
        horizon=horizon,
        target_columns=tcols,
        target_mode=str(target_mode),
        forecast_arch=str(forecast_arch),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        device=device,
        seed=seed,
        model_out_path=model_out,
        prefer_exchange=exchange,
        early_stop_patience=int(early_stop_patience),
        lr_plateau_patience=int(lr_plateau_patience),
        lr_factor=float(lr_factor),
        min_lr=float(min_lr),
    )
    report = train_transformer_forecaster(params)
    typer.echo(json.dumps(report, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))


@app.command()
def tune(
    pair: str = typer.Option("BTC/USDT:USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    window: int = typer.Option(128),
    timesteps: int = typer.Option(300_000),
    arch: str = typer.Option("transformer_hybrid"),
    device: str = typer.Option("cuda"),
    # Search space toggles
    search_space: str = typer.Option("default", help="default|minimal|wide"),
    sampler: str = typer.Option("tpe", help="tpe|random|grid"),
    n_trials: int = typer.Option(30, help="Number of trials (ignored for pure grid if grid > n_trials)"),
    seed: int = typer.Option(42),
    extra_timeframes: str = typer.Option("", help="Optional comma-separated HTFs for all trials"),
    optimize_features: bool = typer.Option(True, help="Search feature subset across ALL individual features"),
    max_features: int = typer.Option(0, help="Optional cap on number of selected features (0 = no cap)"),
    # Search ranges (overrides defaults when provided)
    windows_list: str = typer.Option("", help="Comma-separated window sizes to search; empty uses defaults"),
    min_hold_bars_list: str = typer.Option("", help="Comma-separated min-hold values to search"),
    cooldown_bars_list: str = typer.Option("", help="Comma-separated cooldown values to search"),
    idle_penalty_range: str = typer.Option("0.0,0.05", help="Range for idle_penalty_bps as 'low,high' for TPE/Random; grid uses fixed set"),
    # Fixed env knobs
    turnover_penalty_bps: float = typer.Option(1.0),
    min_hold_bars: int = typer.Option(2),
    cooldown_bars: int = typer.Option(0),
    # Eval and early stop
    eval_max_steps: int = typer.Option(5000),
    eval_freq: int = typer.Option(50000, help="Evaluate every N steps; enable >0 for early stopping"),
    early_stop_metric: str = typer.Option("sharpe", help="sharpe|final_equity|max_drawdown"),
    early_stop_patience: int = typer.Option(3),
    early_stop_min_delta: float = typer.Option(0.0),
    early_stop_degrade_ratio: float = typer.Option(0.0),
    # Auto backtest
    auto_backtest: bool = typer.Option(True, help="Run freqtrade backtest per model"),
    backtest_timerange: str = typer.Option("20250101-", help="YYYYMMDD-YYYYMMDD for freqtrade backtests; empty uses training eval slice"),
    backtest_exchange: str = typer.Option("bybit"),
    # Validation slice after training
    eval_timerange: str = typer.Option("20240101-20250101", help="Timerange for post-train validation report"),
    # Auto-download datasets
    autofetch: bool = typer.Option(True, help="Auto-download missing datasets (1h,4h,1d,1w)"),
    timerange: str = typer.Option("20190101-", help="Timerange for auto-download"),
    exchange: str = typer.Option("bybit", help="Exchange name for dataset download and preference"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "optuna")),
):
    """Hyperparameter tuning using Optuna (TPE=Bayesian, Random, or Grid).

    Optimizes PPO hyperparams and environment shaping. Stores per-trial artifacts and CSV.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler, RandomSampler, GridSampler
    except Exception as e:
        raise RuntimeError("Optuna not installed. Run: pip install -r requirements.txt") from e

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tune_dir = Path(outdir) / ts
    tune_dir.mkdir(parents=True, exist_ok=True)
    results_csv = tune_dir / "results.csv"
    fields = [
        "trial_number","value","model_path","feature_mode","features_used","window","min_hold_bars","cooldown_bars",
        "position_penalty_bps","loss_hold_penalty_bps","cvar_alpha","cvar_coef","max_position_bars",
        "turnover_penalty_bps","extra_timeframes","reward_type","ent_coef","learning_rate","n_steps","batch_size","fee_bps","slippage_bps","seed",
        "eval_timerange","backtest_timerange","exchange",
        "final_equity","sharpe","max_drawdown","time_in_position_frac",
        "bt_total_profit_pct","bt_total_profit_abs","bt_total_trades","bt_win_rate","bt_profit_factor",
        "bt_zip","bt_meta","bt_run_id"
    ]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    etf_list = _parse_list(extra_timeframes, str) if extra_timeframes else []

    # Build candidate sets for extra timeframes exploration.
    # If extra_timeframes contains ';', treat it as semicolon-separated options of comma lists.
    # If empty, use sensible defaults. Otherwise, use provided value as a single fixed option.
    if extra_timeframes and ";" in extra_timeframes:
        etf_candidates_str = [
            ",".join([t.strip().lower() for t in opt.split(",") if t.strip()])
            for opt in extra_timeframes.split(";") if opt.strip()
        ]
    elif extra_timeframes:
        etf_candidates_str = [
            ",".join([t.strip().lower() for t in extra_timeframes.split(",") if t.strip()])
        ]
    else:
        etf_candidates_str = ["", "4h", "1d", "4h,1d", "4h,1d,1w"]

    # Ensure datasets exist if requested
    if autofetch:
        tfs = sorted(set([timeframe, "1h", "4h", "1d", "1w"]))
        for tf in tfs:
            try:
                _ = _ensure_dataset(userdir, pair, tf, exchange=exchange, timerange=timerange)
            except Exception as e:
                typer.echo(f"Auto-download failed for {pair} {tf}: {e}")

    # Candidate lists
    window_candidates = _parse_list(windows_list, int) if windows_list else [64, 96, 128, 192, 256, 384]
    min_hold_candidates = _parse_list(min_hold_bars_list, int) if min_hold_bars_list else [0, 1, 2, 3, 5, 8]
    cooldown_candidates = _parse_list(cooldown_bars_list, int) if cooldown_bars_list else [0, 1, 2, 3, 5]

    # Define search spaces
    def suggest_space(trial):
        # Core PPO
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.2, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

        # Reward/env shaping
        reward_type = trial.suggest_categorical("reward_type", ["vol_scaled", "sharpe_proxy", "raw"]) if search_space != "minimal" else trial.suggest_categorical("reward_type", ["vol_scaled", "raw"]) 
        # Fees/slippage are fixed for realism (exchange fee ~6 bps; slippage median ~5 bps)
        fee_bps = 6.0
        slippage_bps = 5.0

        position_penalty_bps = trial.suggest_float("position_penalty_bps", 0.0, 5.0) if search_space != "minimal" else 0.0
        loss_hold_penalty_bps = trial.suggest_float("loss_hold_penalty_bps", 0.0, 5.0) if search_space == "wide" else 0.0
        cvar_alpha = trial.suggest_categorical("cvar_alpha", [0.0, 0.01, 0.05]) if search_space != "minimal" else 0.0
        cvar_coef = trial.suggest_float("cvar_coef", 0.0, 2.0) if search_space == "wide" else 0.0
        max_position_bars = trial.suggest_categorical("max_position_bars", [0, 24, 48, 96]) if search_space != "minimal" else 0
        # Idle penalty encourages not staying flat forever (drives trades)
        try:
            _lo, _hi = [float(x.strip()) for x in idle_penalty_range.split(',')[:2]]
            if _lo > _hi:
                _lo, _hi = _hi, _lo
        except Exception:
            _lo, _hi = 0.0, 0.05
        idle_penalty_bps = trial.suggest_float("idle_penalty_bps", _lo, _hi) if search_space != "minimal" else 0.02

        # Features / window
        feature_mode = trial.suggest_categorical("feature_mode", ["basic", "full"]) if search_space != "minimal" else "basic"
        win = trial.suggest_categorical("window", window_candidates) if search_space != "minimal" else window
        min_hold = trial.suggest_categorical("min_hold_bars", min_hold_candidates) if search_space != "minimal" else min_hold_bars
        cooldown = trial.suggest_categorical("cooldown_bars", cooldown_candidates) if search_space != "minimal" else cooldown_bars

        # Extra timeframes exploration (string like "4h,1d" or "")
        extra_tfs_str = trial.suggest_categorical("extra_timeframes", etf_candidates_str)

        cfg = {
            "ent_coef": ent_coef,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "reward_type": reward_type,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "position_penalty_bps": position_penalty_bps,
            "loss_hold_penalty_bps": loss_hold_penalty_bps,
            "cvar_alpha": cvar_alpha,
            "cvar_coef": cvar_coef,
            "max_position_bars": max_position_bars,
            "idle_penalty_bps": idle_penalty_bps,
            "feature_mode": feature_mode,
            "window": win,
            "min_hold_bars": min_hold,
            "cooldown_bars": cooldown,
            "extra_timeframes": extra_tfs_str,
        }

        # Per-feature toggles (optional): derive full feature list from features.py documentation
        if optimize_features:
            # Full superset in features.py (excluding OHLCV and HTF prefixes which are included by extra_timeframes)
            base_feature_list = [
                # Existing
                "logret","hl_range","upper_wick","lower_wick","rsi","vol_z","atr","ret_std_14","hurst","tail_alpha","risk_gate","ema_fast_ratio","ema_slow_ratio",
                # New trend/momentum
                "sma_ratio","ema_cross_angle","lr_slope_90","macd_line","macd_signal","stoch_k","stoch_d","roc_10",
                # Vol/risk
                "bb_width_20","donchian_width_20","true_range_pct","vol_skewness_30","volatility_regime",
                # Volume
                "obv_z","volume_delta_z","vwap_ratio_20",
                # Microstructure
                "candle_body_frac","upper_shadow_frac","lower_shadow_frac","candle_trend_persistence","kurtosis_rolling_100",
                # Stats/Fractal
                "dfa_exponent_64","entropy_return_64",
                # Risk mgmt
                "drawdown_z_64","expected_shortfall_0_05_128",
                # Long-horizon MAs (daily/weekly resamples available in features)
                "MA365D","MA200D","MA50D","MA20W",
            ]
            chosen: list[str] = []
            for fname in base_feature_list:
                use = trial.suggest_categorical(f"f_{fname}", [True, False])
                if use:
                    chosen.append(fname)
            if max_features and len(chosen) > max_features:
                chosen = chosen[:max_features]
            # Always ensure non-empty; fallback to a couple core features
            if not chosen:
                chosen = ["logret","rsi","atr"]
            cfg["feature_whitelist"] = chosen
            # Ensure unified superset mode so whitelist columns exist
            cfg["feature_mode"] = "full"
        return cfg

    # Grid definitions for GridSampler
    grid = {
        "ent_coef": [0.01, 0.02, 0.05, 0.1],
        "learning_rate": [3e-4, 5e-4, 1e-4],
        "n_steps": [1024, 2048],
        "batch_size": [256, 512],
        "reward_type": ["vol_scaled", "sharpe_proxy", "raw"],
        "position_penalty_bps": [0.0, 1.0],
        "loss_hold_penalty_bps": [0.0, 1.0],
        "cvar_alpha": [0.0, 0.05],
        "cvar_coef": [0.0, 0.5],
        "max_position_bars": [0, 48],
        "idle_penalty_bps": [0.0, 0.02, 0.05],
        "feature_mode": ["basic", "full"],
        "window": window_candidates if windows_list else [window],
        "min_hold_bars": min_hold_candidates if min_hold_bars_list else [min_hold_bars],
        "cooldown_bars": cooldown_candidates if cooldown_bars_list else [cooldown_bars],
        "extra_timeframes": etf_candidates_str,
    }

    if sampler.lower() == "tpe":
        smp = TPESampler(seed=seed)
    elif sampler.lower() == "random":
        smp = RandomSampler(seed=seed)
    elif sampler.lower() == "grid":
        smp = GridSampler(grid)
    else:
        raise typer.BadParameter("sampler must be tpe|random|grid")

    def objective(trial: "optuna.trial.Trial") -> float:
        cfg = suggest_space(trial) if not isinstance(smp, GridSampler) else {k: trial.suggest_categorical(k, v) for k, v in grid.items()}

        fee_fixed = 6.0
        slip_fixed = 5.0
        etf_tag = (cfg.get('extra_timeframes') or 'none').replace(',', '-')
        tag = (
            f"{arch}_fm-{cfg['feature_mode']}_win-{cfg['window']}"
            f"_mh-{cfg['min_hold_bars']}_cd-{cfg['cooldown_bars']}"
            f"_rt-{cfg['reward_type']}_ec-{cfg['ent_coef']}_lr-{cfg['learning_rate']}"
            f"_ns-{cfg['n_steps']}_bs-{cfg['batch_size']}_fee-{fee_fixed}_slip-{slip_fixed}_etf-{etf_tag}_seed-{seed}"
        )
        model_path = str(tune_dir / f"{tag}.zip")

        params = TrainParams(
            userdir=userdir,
            pair=pair,
            timeframe=timeframe,
            window=int(cfg["window"]),
            total_timesteps=int(timesteps),
            model_out_path=model_path,
            arch=arch,
            device=device,
            prefer_exchange=exchange,
            fee_bps=fee_fixed,
            slippage_bps=slip_fixed,
            # override with tuned idle penalty when present (kept default 0.0 for safety)
            idle_penalty_bps=float(cfg.get("idle_penalty_bps", 0.0)),
            reward_type=str(cfg["reward_type"]),
            vol_lookback=20,
            turnover_penalty_bps=float(turnover_penalty_bps),
            dd_penalty=0.05 if cfg["reward_type"] == "sharpe_proxy" else 0.0,
            position_penalty_bps=float(cfg["position_penalty_bps"]),
            loss_hold_penalty_bps=float(cfg["loss_hold_penalty_bps"]),
            cvar_alpha=float(cfg["cvar_alpha"]),
            cvar_coef=float(cfg["cvar_coef"]),
            max_position_bars=int(cfg["max_position_bars"]),
            min_hold_bars=int(cfg["min_hold_bars"]),
            cooldown_bars=int(cfg["cooldown_bars"]),
            random_reset=True,
            episode_max_steps=4096,
            feature_mode=str(cfg["feature_mode"]),
            basic_lookback=128 if str(cfg["feature_mode"]) == "basic" else 64,
            extra_timeframes=_parse_list(str(cfg.get("extra_timeframes", "")), str) or None,
            feature_groups=list(cfg.get("feature_groups", []) or []),
            feature_whitelist=list(cfg.get("feature_whitelist", []) or []),
            eval_freq=eval_freq,
            n_eval_episodes=1,
            eval_max_steps=eval_max_steps,
            eval_log_path=str(tune_dir / "eval_log.csv"),
            seed=seed,
            ent_coef=float(cfg["ent_coef"]),
            learning_rate=float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            n_epochs=10,
            early_stop_metric=(early_stop_metric or None),
            early_stop_patience=int(early_stop_patience),
            early_stop_min_delta=float(early_stop_min_delta),
            early_stop_degrade_ratio=float(early_stop_degrade_ratio),
        )

        try:
            _ = train_ppo_from_freqtrade_data(params)
            report = validate_trained_model(params, max_steps=eval_max_steps, deterministic=True, timerange=eval_timerange)
            bt_metrics: Dict[str, Any] = {}
            if auto_backtest:
                bt_metrics = _run_freqtrade_backtest(
                    userdir=userdir,
                    pair=pair,
                    timeframe=timeframe,
                    exchange=backtest_exchange,
                    timerange=(backtest_timerange or (params.timerange or "")) or "",
                    model_path=model_path,
                    window=int(cfg["window"]),
                    device=device,
                    min_hold_bars=int(cfg["min_hold_bars"]),
                    cooldown_bars=int(cfg["cooldown_bars"]),
                )
        except Exception as e:
            typer.echo(f"Trial failed: {tag} -> {e}")
            raise

        # Choose objective: maximize Sharpe (Optuna minimizes by default -> return -sharpe)
        sharpe = _safe_float(report.get("sharpe"))
        value = sharpe if sharpe == sharpe else float("-inf")  # NaN guard

        row = {
            "trial_number": trial.number,
            "value": float(value),
            "model_path": model_path,
            "feature_mode": str(cfg["feature_mode"]),
            "features_used": ",".join(list(cfg.get("feature_whitelist", []))) if cfg.get("feature_whitelist") else "",
            "window": int(cfg["window"]),
            "min_hold_bars": int(cfg["min_hold_bars"]),
            "cooldown_bars": int(cfg["cooldown_bars"]),
            "position_penalty_bps": _safe_float(cfg["position_penalty_bps"]),
            "loss_hold_penalty_bps": _safe_float(cfg["loss_hold_penalty_bps"]),
            "cvar_alpha": _safe_float(cfg["cvar_alpha"]),
            "cvar_coef": _safe_float(cfg["cvar_coef"]),
            "max_position_bars": _safe_int(cfg["max_position_bars"]),
            "turnover_penalty_bps": _safe_float(turnover_penalty_bps),
            "extra_timeframes": str(cfg.get("extra_timeframes", "")),
            "reward_type": str(cfg["reward_type"]),
            "ent_coef": _safe_float(cfg["ent_coef"]),
            "learning_rate": _safe_float(cfg["learning_rate"]),
            "n_steps": _safe_int(cfg["n_steps"]),
            "batch_size": _safe_int(cfg["batch_size"]),
            "fee_bps": _safe_float(fee_fixed),
            "slippage_bps": _safe_float(slip_fixed),
            "seed": _safe_int(seed),
            "eval_timerange": str(eval_timerange),
            "backtest_timerange": str(backtest_timerange or ""),
            "exchange": str(exchange),
            "final_equity": _safe_float(report.get("final_equity")),
            "sharpe": _safe_float(report.get("sharpe")),
            "max_drawdown": _safe_float(report.get("max_drawdown")),
            "time_in_position_frac": _safe_float(report.get("time_in_position_frac")),
            "bt_total_profit_pct": _safe_float(bt_metrics.get("bt_total_profit_pct")) if auto_backtest else float("nan"),
            "bt_total_profit_abs": _safe_float(bt_metrics.get("bt_total_profit_abs")) if auto_backtest else float("nan"),
            "bt_total_trades": _safe_int(bt_metrics.get("bt_total_trades")) if auto_backtest else 0,
            "bt_win_rate": _safe_float(bt_metrics.get("bt_win_rate")) if auto_backtest else float("nan"),
            "bt_profit_factor": _safe_float(bt_metrics.get("bt_profit_factor")) if auto_backtest else float("nan"),
            "bt_zip": str(bt_metrics.get("bt_zip", "")) if auto_backtest else "",
            "bt_meta": str(bt_metrics.get("bt_meta", "")) if auto_backtest else "",
            "bt_run_id": str(bt_metrics.get("bt_run_id", "")) if auto_backtest else "",
        }
        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)

        # Optuna maximizes when direction='maximize'
        return value

    study = optuna.create_study(direction="maximize", sampler=smp)
    typer.echo(f"Starting Optuna study: {study.study_name} -> dir {tune_dir}")
    if isinstance(smp, GridSampler):
        grid_size = 1
        for v in grid.values():
            grid_size *= max(1, len(v))
        study.optimize(objective, n_trials=grid_size)
    else:
        study.optimize(objective, n_trials=n_trials)

    # Save best params
    with open(tune_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    with open(tune_dir / "best_value.txt", "w") as f:
        f.write(str(study.best_value))
    typer.echo(f"Best value: {study.best_value}\nBest params: {json.dumps(study.best_params)}")

@app.command()
def xgb_tune(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    # Data
    timerange: str = typer.Option("20190101-", help="YYYYMMDD-YYYYMMDD slice for dataset"),
    exchange: str = typer.Option("bybit", help="Prefer this exchange's dataset when multiple exist"),
    eval_timerange: str = typer.Option("", help="Optional explicit validation slice YYYYMMDD-YYYYMMDD; train is before start"),
    train_ratio: float = typer.Option(0.8, help="Train fraction if eval_timerange not set"),
    feature_mode: str = typer.Option("full", help="full|basic for feature generation"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("", help="Optional HTFs, e.g., '4H,1D' or multiple options with ';'"),
    horizon: int = typer.Option(1, help="Predict sum of next H log-returns (H>=1)"),
    # Search
    sampler: str = typer.Option("tpe", help="tpe|random|grid"),
    n_trials: int = typer.Option(40, help="Number of trials (grid uses its own size)"),
    seed: int = typer.Option(42),
    optimize_features: bool = typer.Option(True, help="Search per-feature subset from hand-made features"),
    max_features: int = typer.Option(0, help="Cap number of selected features (0=no cap)"),
    ic_metric: str = typer.Option("spearman", help="IC metric: spearman|pearson"),
    n_jobs: int = typer.Option(0, help="Threads for XGBoost; 0=all cores"),
    xgb_device: str = typer.Option("cpu", help="XGBoost device: auto|cpu|cuda"),
    autofetch: bool = typer.Option(True, help="Auto-download dataset if missing"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_optuna")),
):
    """Optuna tuning for XGBoost on hand-made features, optimizing validation IC.

    Trains a fast GBT regressor to predict next-H log return. Objective is IC on
    a held-out validation set (Spearman by default). Also searches a feature subset.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler, RandomSampler, GridSampler
    except Exception as e:
        raise RuntimeError("Optuna not installed. Run: pip install -r requirements.txt") from e
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing deps. Install xgboost, numpy, pandas.") from e

    # Resolve optional extra timeframe candidates (mirrors PPO tuner)
    etf_list_fixed = _parse_list(extra_timeframes, str) if extra_timeframes else []
    if extra_timeframes and ";" in extra_timeframes:
        etf_candidates_str = [
            ",".join([t.strip().lower() for t in opt.split(",") if t.strip()])
            for opt in extra_timeframes.split(";") if opt.strip()
        ]
    elif extra_timeframes:
        etf_candidates_str = [
            ",".join([t.strip().lower() for t in extra_timeframes.split(",") if t.strip()])
        ]
    else:
        etf_candidates_str = ["", "4h", "1d", "4h,1d", "4h,1d,1w"]

    # Ensure dataset exists (optional)
    if autofetch:
        try:
            _ = _ensure_dataset(userdir, pair, timeframe, exchange=exchange, timerange=timerange)
        except Exception as e:
            typer.echo(f"Auto-download failed for {pair} {timeframe}: {e}")

    # Locate and load
    data_path = _find_data_file_internal(userdir, pair, timeframe, prefer_exchange=exchange)
    if not data_path:
        raise FileNotFoundError("No dataset found. Run download or set correct userdir/pair/timeframe.")
    raw = _load_ohlcv_internal(data_path)

    # Helper: timerange slicing
    def _slice_df(df, tr: str | None):
        if not tr:
            return df
        try:
            start_str, end_str = tr.split('-', 1)
            import pandas as _pd
            start = _pd.to_datetime(start_str) if start_str else None
            end = _pd.to_datetime(end_str) if end_str else None
            if isinstance(df.index, _pd.DatetimeIndex):
                idx = df.index
                try:
                    idx_cmp = idx.tz_convert(None) if idx.tz is not None else idx
                except Exception:
                    idx_cmp = idx.tz_localize(None) if getattr(idx, 'tz', None) is not None else idx
                import pandas as _pd2
                mask = _pd2.Series(True, index=idx)
                if start is not None:
                    mask &= idx_cmp >= _pd2.to_datetime(start)
                if end is not None:
                    mask &= idx_cmp <= _pd2.to_datetime(end)
                return df.loc[mask]
        except Exception:
            return df
        return df

    raw = _slice_df(raw, timerange)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tune_dir = Path(outdir) / ts
    tune_dir.mkdir(parents=True, exist_ok=True)
    results_csv = tune_dir / "results.csv"
    fields = [
        "trial_number","value","ic_spearman","ic_pearson","mse","acc_dir",
        "features_used","feature_mode","extra_timeframes","horizon","seed",
        "learning_rate","max_depth","min_child_weight","subsample","colsample_bytree","reg_alpha","reg_lambda","n_estimators",
        "eval_start","eval_end","model_path","device"
    ]
    with open(results_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    # Candidate per-feature whitelist (hand-made features from features.py)
    base_feature_list = [
        "logret","hl_range","upper_wick","lower_wick","rsi","vol_z","atr","ret_std_14","hurst","tail_alpha","risk_gate","ema_fast_ratio","ema_slow_ratio",
        "sma_ratio","ema_cross_angle","lr_slope_90","macd_line","macd_signal","stoch_k","stoch_d","roc_10",
        "bb_width_20","donchian_width_20","true_range_pct","vol_skewness_30","volatility_regime",
        "obv_z","volume_delta_z","vwap_ratio_20",
        "candle_body_frac","upper_shadow_frac","lower_shadow_frac","candle_trend_persistence","kurtosis_rolling_100",
        "dfa_exponent_64","entropy_return_64",
        "drawdown_z_64","expected_shortfall_0_05_128",
        "MA365D","MA200D","MA50D","MA20W",
    ]

    # Build sampler
    xgb_grid = None
    if sampler.lower() == "tpe":
        smp = TPESampler(seed=seed)
    elif sampler.lower() == "random":
        smp = RandomSampler(seed=seed)
    elif sampler.lower() == "grid":
        xgb_grid = {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1.0, 3.0, 5.0],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "reg_alpha": [0.0, 0.001, 0.01],
            "reg_lambda": [1.0, 3.0, 5.0],
            "n_estimators": [400, 800, 1200],
            "extra_timeframes": etf_candidates_str,
        }
        smp = GridSampler(xgb_grid)
    else:
        raise typer.BadParameter("sampler must be tpe|random|grid")

    def suggest_space(trial):
        cfg = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
            "extra_timeframes": trial.suggest_categorical("extra_timeframes", etf_candidates_str),
        }
        chosen = []
        if optimize_features:
            for fname in base_feature_list:
                use = trial.suggest_categorical(f"f_{fname}", [True, False])
                if use:
                    chosen.append(fname)
            if max_features and len(chosen) > max_features:
                chosen = chosen[:max_features]
            if not chosen:
                chosen = ["logret","rsi","atr"]
            cfg["feature_whitelist"] = chosen
        return cfg

    best = {"value": float("-inf"), "path": ""}

    def _ic(y_true, y_pred, method: str = "spearman") -> float:
        try:
            import pandas as _pd
            s1 = _pd.Series(y_true)
            s2 = _pd.Series(y_pred)
            if method == "pearson":
                v = float(s1.corr(s2, method="pearson"))
            else:
                v = float(s1.corr(s2, method="spearman"))
            if v != v:
                return float("-inf")
            return v
        except Exception:
            return float("-inf")

    study = optuna.create_study(direction="maximize", sampler=smp)
    typer.echo(f"Starting XGB Optuna study: {study.study_name} -> dir {tune_dir}")

    eval_start = eval_end = None
    if eval_timerange:
        try:
            import pandas as _pd
            s, e = eval_timerange.split('-', 1)
            eval_start = _pd.to_datetime(s) if s else None
            eval_end = _pd.to_datetime(e) if e else None
        except Exception:
            eval_start = eval_end = None

    def objective(trial):
        # Resolve device selection per-trial (static across trials)
        dev_opt = str(xgb_device).strip().lower()
        if dev_opt == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                dev = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        elif dev_opt in ("cpu", "cuda"):
            dev = dev_opt
        else:
            dev = "cpu"
        
        # Verify CUDA availability if requested
        if dev == "cuda":
            try:
                import subprocess
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    typer.echo(f"WARNING: CUDA requested but nvidia-smi failed. Falling back to CPU.")
                    dev = "cpu"
                else:
                    typer.echo(f"CUDA detected: using GPU for XGBoost trial {trial.number}")
            except Exception:
                typer.echo(f"WARNING: Could not verify CUDA. Attempting GPU anyway...")
        
        typer.echo(f"Trial {trial.number}: device={dev}")
        cfg = suggest_space(trial) if not isinstance(smp, GridSampler) else {k: trial.suggest_categorical(k, v) for k, v in (xgb_grid or {}).items()}
        etf = [t.strip() for t in str(cfg.get("extra_timeframes", "")).split(",") if t.strip()]
        feature_cols = None
        if optimize_features:
            keep = ["open","high","low","close","volume"] + list(cfg.get("feature_whitelist", []))
            feature_cols = keep
            fm = "full"
        else:
            fm = feature_mode
        from rl_lib.features import make_features as _make_features
        feats = _make_features(
            raw,
            feature_columns=feature_cols,
            mode=fm,
            basic_lookback=int(basic_lookback),
            extra_timeframes=([s.upper() for s in etf] if etf else (etf_list_fixed or None)),
        )
        c = feats["close"].astype(float).values if "close" in feats.columns else raw["close"].astype(float).values
        if horizon <= 0:
            raise typer.BadParameter("horizon must be >= 1")
        import numpy as _np
        import pandas as _pd
        logp = _pd.Series(_np.log(c + 1e-12), index=feats.index)
        y = (logp.shift(-int(horizon)) - logp).to_numpy()
        valid_len = len(feats) - int(horizon)
        if valid_len <= 100:
            return float("-inf")
        X = feats.iloc[:valid_len, :].copy()
        y = y[:valid_len].astype(float)
        if eval_start is not None or eval_end is not None:
            if not isinstance(X.index, _pd.DatetimeIndex):
                return float("-inf")
            idx = X.index
            try:
                idx_cmp = idx.tz_convert(None) if idx.tz is not None else idx
            except Exception:
                idx_cmp = idx.tz_localize(None) if getattr(idx, 'tz', None) is not None else idx
            mask_val = _pd.Series(True, index=idx)
            if eval_start is not None:
                mask_val &= idx_cmp >= _pd.to_datetime(eval_start)
            if eval_end is not None:
                mask_val &= idx_cmp <= _pd.to_datetime(eval_end)
            X_val = X.loc[mask_val].copy()
            y_val = y[mask_val.to_numpy()]
            X_train = X.loc[~mask_val].copy()
            y_train = y[~mask_val.to_numpy()]
        else:
            cut = int(max(10, min(len(X) - 10, int(len(X) * float(train_ratio)))))
            X_train = X.iloc[:cut, :].copy()
            y_train = y[:cut]
            X_val = X.iloc[cut:, :].copy()
            y_val = y[cut:]
        if X_train.empty or X_val.empty:
            return float("-inf")

        # XGBoost 2.x: use device="cuda" with tree_method="hist" (not gpu_hist)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",  # Always hist; device parameter controls GPU usage
            random_state=int(seed),
            n_jobs=(int(n_jobs) if int(n_jobs) > 0 else -1),
            device=dev,
            learning_rate=float(cfg["learning_rate"]),
            max_depth=int(cfg["max_depth"]),
            min_child_weight=float(cfg["min_child_weight"]),
            subsample=float(cfg["subsample"]),
            colsample_bytree=float(cfg["colsample_bytree"]),
            reg_alpha=float(cfg["reg_alpha"]),
            reg_lambda=float(cfg["reg_lambda"]),
            n_estimators=int(cfg["n_estimators"]),
        )
        try:
            # XGBoost 2.x sklearn API compatibility
            # For GPU: convert to appropriate data structures
            if dev == "cuda":
                try:
                    import cupy as cp
                    X_train_gpu = cp.asarray(X_train.values, dtype=cp.float32)
                    y_train_gpu = cp.asarray(y_train, dtype=cp.float32)
                    X_val_gpu = cp.asarray(X_val.values, dtype=cp.float32)
                    y_val_gpu = cp.asarray(y_val, dtype=cp.float32)
                    
                    model.fit(
                        X_train_gpu, y_train_gpu,
                        eval_set=[(X_val_gpu, y_val_gpu)],
                        verbose=False,
                    )
                    y_pred = cp.asnumpy(model.predict(X_val_gpu))
                    # Check GPU memory usage
                    gpu_mem = cp.cuda.Device().mem_info
                    used_mb = (gpu_mem[1] - gpu_mem[0]) / (1024**2)
                    typer.echo(f"Trial {trial.number}: GPU training complete, using {used_mb:.0f}MB VRAM")
                except ImportError:
                    typer.echo(f"Trial {trial.number}: CuPy not available, using numpy arrays on GPU")
                    model.fit(
                        X_train.values.astype('float32'), y_train.astype('float32'),
                        eval_set=[(X_val.values.astype('float32'), y_val.astype('float32'))],
                        verbose=False,
                    )
                    y_pred = model.predict(X_val.values.astype('float32'))
            else:
                model.fit(
                    X_train.values, y_train,
                    eval_set=[(X_val.values, y_val)],
                    verbose=False,
                )
                y_pred = model.predict(X_val.values)
        except Exception as e:
            typer.echo(f"Trial fit failed: {e}")
            return float("-inf")
        ic_s = _ic(y_val, y_pred, method="spearman")
        ic_p = _ic(y_val, y_pred, method="pearson")
        import numpy as _np2
        mse = float(_np2.mean((y_val - y_pred) ** 2)) if len(y_val) == len(y_pred) else float("inf")
        acc = float(_np2.mean(_np2.sign(y_val) == _np2.sign(y_pred))) if len(y_val) == len(y_pred) else float("nan")
        value = ic_s if str(ic_metric).lower() == "spearman" else ic_p
        if value != value or value == float("-inf"):
            value = float("-inf")

        tag = (
            f"lr-{cfg['learning_rate']}_md-{cfg['max_depth']}_mcw-{cfg['min_child_weight']}_"
            f"ss-{cfg['subsample']}_cs-{cfg['colsample_bytree']}_ra-{cfg['reg_alpha']}_rl-{cfg['reg_lambda']}_"
            f"ne-{cfg['n_estimators']}_H-{int(horizon)}"
        )
        model_path = str(tune_dir / f"xgb_{tag}.json")
        try:
            if value > float(best.get("value", float("-inf"))):
                model.save_model(model_path)
                feat_cols_path = str(Path(model_path).with_suffix("").as_posix()) + "_feature_columns.json"
                with open(feat_cols_path, "w") as f:
                    json.dump(list(X.columns), f)
                meta = {
                    "ic_spearman": float(ic_s),
                    "ic_pearson": float(ic_p),
                    "mse": float(mse),
                    "acc_dir": float(acc),
                    "horizon": int(horizon),
                    "extra_timeframes": etf,
                }
                with open(str(Path(model_path).with_suffix("").as_posix()) + "_meta.json", "w") as f:
                    json.dump(meta, f, indent=2)
                best["value"] = float(value)
                best["path"] = model_path
        except Exception:
            pass

        row = {
            "trial_number": trial.number,
            "value": float(value),
            "ic_spearman": float(ic_s),
            "ic_pearson": float(ic_p),
            "mse": float(mse),
            "acc_dir": float(acc),
            "features_used": ",".join(list(cfg.get("feature_whitelist", []))) if cfg.get("feature_whitelist") else "",
            "feature_mode": "full" if optimize_features else str(feature_mode),
            "extra_timeframes": ",".join(etf) if etf else ",".join(etf_list_fixed) if etf_list_fixed else "",
            "horizon": int(horizon),
            "seed": int(seed),
            "learning_rate": float(cfg["learning_rate"]),
            "max_depth": int(cfg["max_depth"]),
            "min_child_weight": float(cfg["min_child_weight"]),
            "subsample": float(cfg["subsample"]),
            "colsample_bytree": float(cfg["colsample_bytree"]),
            "reg_alpha": float(cfg["reg_alpha"]),
            "reg_lambda": float(cfg["reg_lambda"]),
            "n_estimators": int(cfg["n_estimators"]),
            "eval_start": str(eval_start) if eval_start is not None else "",
            "eval_end": str(eval_end) if eval_end is not None else "",
            "model_path": best.get("path", ""),
            "device": dev,
        }
        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)
        return float(value)

    if isinstance(smp, GridSampler):
        grid_size = 1
        for v in (xgb_grid or {}).values():
            grid_size *= max(1, len(v))
        study.optimize(objective, n_trials=grid_size)
    else:
        study.optimize(objective, n_trials=n_trials)

    with open(tune_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    with open(tune_dir / "best_value.txt", "w") as f:
        f.write(str(study.best_value))
    typer.echo(f"Best value: {study.best_value}\nBest params: {json.dumps(study.best_params)}\nBest model: {best.get('path','')}")

@app.command()
def xgb_impulse_tune(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    # Data
    timerange: str = typer.Option("20190101-", help="YYYYMMDD-YYYYMMDD slice"),
    exchange: str = typer.Option("bybit"),
    eval_timerange: str = typer.Option("", help="Validation slice YYYYMMDD-YYYYMMDD"),
    train_ratio: float = typer.Option(0.8, help="Train fraction if no eval_timerange"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("", help="'4H,1D' or multiple sets via ';'"),
    horizon: int = typer.Option(8, help="Impulse window H (bars)"),
    tune_horizon: bool = typer.Option(False, help="Enable Optuna to tune horizon in [horizon_min,horizon_max]"),
    horizon_min: int = typer.Option(1),
    horizon_max: int = typer.Option(24),
    label_mode: str = typer.Option("vol", help="vol|abs: vol=alpha*sigma*sqrt(H) threshold; abs=|ret|>bps"),
    alpha_up: float = typer.Option(2.0, help="Vol-scaled up threshold multiplier (σ)"),
    alpha_dn: float = typer.Option(2.0, help="Vol-scaled down threshold multiplier (σ)"),
    vol_lookback: int = typer.Option(256, help="Rolling std lookback for σ (bars)"),
    thr_up_bps: float = typer.Option(30.0, help="(abs mode) Up-spike threshold in bps of max fwd logret"),
    thr_dn_bps: float = typer.Option(30.0, help="(abs mode) Down-spike threshold in bps of min fwd logret"),
    # Search
    sampler: str = typer.Option("tpe", help="tpe|random|grid"),
    n_trials: int = typer.Option(40),
    seed: int = typer.Option(42),
    optimize_features: bool = typer.Option(True),
    max_features: int = typer.Option(0),
    n_jobs: int = typer.Option(0),
    oversample_pos: float = typer.Option(0.0, help="Duplicate positive class by this factor (0=off)"),
    xgb_device: str = typer.Option("auto", help="auto|cpu|cuda"),
    autofetch: bool = typer.Option(True),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "xgb_impulse")),
):
    """Optuna tuning for impulse classifiers (up/down spikes), optimizing AUPRC.

    Trains two binary XGBoost classifiers for spike_up and spike_down events based on
    forward max/min log returns over H bars exceeding thresholds (bps).
    """
    try:
        import optuna
        from optuna.samplers import TPESampler, RandomSampler, GridSampler
    except Exception as e:
        raise RuntimeError("Optuna not installed. Run: pip install -r requirements.txt") from e
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import xgboost as xgb  # type: ignore
        from sklearn.metrics import average_precision_score  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing deps. Install xgboost, numpy, pandas, scikit-learn.") from e

    # Auto data
    if autofetch:
        try:
            _ = _ensure_dataset(userdir, pair, timeframe, exchange=exchange, timerange=timerange)
        except Exception as e:
            typer.echo(f"Auto-download failed for {pair} {timeframe}: {e}")
    data_path = _find_data_file_internal(userdir, pair, timeframe, prefer_exchange=exchange)
    if not data_path:
        raise FileNotFoundError("No dataset found. Run download first.")
    raw = _load_ohlcv_internal(data_path)

    # Extra TF candidates
    if extra_timeframes and ";" in extra_timeframes:
        etf_candidates_str = [
            ",".join([t.strip().lower() for t in opt.split(",") if t.strip()])
            for opt in extra_timeframes.split(";") if opt.strip()
        ]
    elif extra_timeframes:
        etf_candidates_str = [
            ",".join([t.strip().lower() for t in extra_timeframes.split(",") if t.strip()])
        ]
    else:
        etf_candidates_str = ["", "4h", "1d", "4h,1d"]

    # Device resolve
    def _resolve_device() -> str:
        dev_opt = str(xgb_device).strip().lower()
        if dev_opt == "auto":
            try:
                import torch  # type: ignore
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        if dev_opt in ("cpu", "cuda"):
            return dev_opt
        return "cpu"

    # Features superset (same as reg)
    base_feature_list = [
        "logret","hl_range","upper_wick","lower_wick","rsi","vol_z","atr","ret_std_14","hurst","tail_alpha","risk_gate","ema_fast_ratio","ema_slow_ratio",
        "sma_ratio","ema_cross_angle","lr_slope_90","macd_line","macd_signal","stoch_k","stoch_d","roc_10",
        "bb_width_20","donchian_width_20","true_range_pct","vol_skewness_30","volatility_regime",
        "obv_z","volume_delta_z","vwap_ratio_20",
        "candle_body_frac","upper_shadow_frac","lower_shadow_frac","candle_trend_persistence","kurtosis_rolling_100",
        "dfa_exponent_64","entropy_return_64",
        "drawdown_z_64","expected_shortfall_0_05_128",
        "MA365D","MA200D","MA50D","MA20W",
    ]

    # Study out dir
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tune_dir = Path(outdir) / ts
    tune_dir.mkdir(parents=True, exist_ok=True)
    results_csv = tune_dir / "results.csv"
    fields = [
        "trial","value","auprc_up","auprc_down","pos_rate_up","pos_rate_down","features_used","extra_timeframes",
        "learning_rate","max_depth","min_child_weight","subsample","colsample_bytree","reg_alpha","reg_lambda","n_estimators",
        "thr_up_bps","thr_dn_bps","device","model_up","model_down"
    ]
    with open(results_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    # Sampler
    if sampler.lower() == "tpe":
        smp = TPESampler(seed=seed)
    elif sampler.lower() == "random":
        smp = RandomSampler(seed=seed)
    elif sampler.lower() == "grid":
        grid = {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1.0, 3.0, 5.0],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "reg_alpha": [0.0, 0.001, 0.01],
            "reg_lambda": [1.0, 3.0, 5.0],
            "n_estimators": [400, 800, 1200],
            "extra_timeframes": etf_candidates_str,
        }
        smp = GridSampler(grid)
    else:
        raise typer.BadParameter("sampler must be tpe|random|grid")

    def suggest_space(trial):
        cfg = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
            "extra_timeframes": trial.suggest_categorical("extra_timeframes", etf_candidates_str),
        }
        if bool(tune_horizon):
            cfg["horizon"] = int(trial.suggest_int("horizon", int(max(1, horizon_min)), int(max(horizon_min, horizon_max))))
        if optimize_features:
            chosen: list[str] = []
            for fname in base_feature_list:
                use = trial.suggest_categorical(f"f_{fname}", [True, False])
                if use:
                    chosen.append(fname)
            if max_features and len(chosen) > max_features:
                chosen = chosen[:max_features]
            if not chosen:
                chosen = ["logret","rsi","atr"]
            cfg["feature_whitelist"] = chosen
        return cfg

    study = optuna.create_study(direction="maximize", sampler=smp)
    typer.echo(f"Starting Impulse study: {study.study_name} -> dir {tune_dir}")

    # Build eval slice
    eval_start = eval_end = None
    if eval_timerange:
        try:
            s, e = eval_timerange.split('-', 1)
            eval_start = pd.to_datetime(s) if s else None
            eval_end = pd.to_datetime(e) if e else None
        except Exception:
            eval_start = eval_end = None

    best_val = float('-inf')
    best_paths = {"up": "", "down": ""}

    def objective(trial):
        dev = _resolve_device()
        if dev == "cuda":
            typer.echo(f"CUDA detected: using GPU for Impulse trial {trial.number}")

        cfg = suggest_space(trial) if not isinstance(smp, GridSampler) else {k: trial.suggest_categorical(k, v) for k, v in smp.search_space().items()}  # type: ignore[attr-defined]
        etf = [t.strip() for t in str(cfg.get("extra_timeframes", "")).split(",") if t.strip()]
        feature_cols = None
        if optimize_features:
            keep = ["open","high","low","close","volume"] + list(cfg.get("feature_whitelist", []))
            feature_cols = keep
            fm = "full"
        else:
            fm = feature_mode

        # Features
        from rl_lib.features import make_features as _make_features
        feats = _make_features(
            raw,
            feature_columns=feature_cols,
            mode=fm,
            basic_lookback=int(basic_lookback),
            extra_timeframes=([s.upper() for s in etf] if etf else None),
        )

        # Labels from forward window (vol or abs)
        c = feats["close"].astype(float).values if "close" in feats.columns else raw["close"].astype(float).values
        logp = pd.Series(np.log(c + 1e-12), index=feats.index)
        H = int(max(1, int(cfg.get("horizon", horizon))))
        fwd = [logp.shift(-i) - logp for i in range(1, H + 1)]
        fwd_mat = np.vstack([s.to_numpy() for s in fwd])  # (H, T)
        fwd_max = np.nanmax(fwd_mat, axis=0)
        fwd_min = np.nanmin(fwd_mat, axis=0)
        mode_l = str(label_mode).strip().lower()
        if mode_l == "vol":
            # Rolling std of single-bar log returns (causal)
            logret = np.diff(np.log(c + 1e-12), prepend=np.log(c[0] + 1e-12))
            sigma = pd.Series(logret).rolling(int(max(10, vol_lookback)), min_periods=10).std().fillna(0.0).to_numpy()
            # Scale threshold by sqrt(H)
            up_thr_vec = float(alpha_up) * sigma * float(np.sqrt(H))
            dn_thr_vec = float(alpha_dn) * sigma * float(np.sqrt(H))
            y_up = (fwd_max >= up_thr_vec).astype(int)
            y_dn = (fwd_min <= -dn_thr_vec).astype(int)
        else:
            up_thr = float(thr_up_bps) * 1e-4
            dn_thr = float(thr_dn_bps) * 1e-4
            y_up = (fwd_max >= up_thr).astype(int)
            y_dn = (fwd_min <= -dn_thr).astype(int)
        valid_len = len(feats) - H
        if valid_len <= 200:
            return float('-inf')
        X = feats.iloc[:valid_len, :].copy()
        y_up = y_up[:valid_len]
        y_dn = y_dn[:valid_len]

        # Split
        if eval_start is not None or eval_end is not None:
            if not isinstance(X.index, pd.DatetimeIndex):
                return float('-inf')
            idx = X.index
            try:
                idx_cmp = idx.tz_convert(None) if idx.tz is not None else idx
            except Exception:
                idx_cmp = idx.tz_localize(None) if getattr(idx, 'tz', None) is not None else idx
            mask_val = pd.Series(True, index=idx)
            if eval_start is not None:
                mask_val &= idx_cmp >= pd.to_datetime(eval_start)
            if eval_end is not None:
                mask_val &= idx_cmp <= pd.to_datetime(eval_end)
            X_val = X.loc[mask_val].copy()
            y_up_val = y_up[mask_val.to_numpy()]
            y_dn_val = y_dn[mask_val.to_numpy()]
            X_tr = X.loc[~mask_val].copy()
            y_up_tr = y_up[~mask_val.to_numpy()]
            y_dn_tr = y_dn[~mask_val.to_numpy()]
        else:
            cut = int(max(50, min(len(X) - 50, int(len(X) * float(train_ratio)))))
            X_tr = X.iloc[:cut, :].copy(); X_val = X.iloc[cut:, :].copy()
            y_up_tr = y_up[:cut]; y_up_val = y_up[cut:]
            y_dn_tr = y_dn[:cut]; y_dn_val = y_dn[cut:]
        if X_tr.empty or X_val.empty:
            return float('-inf')

        # Scale pos weight to handle imbalance
        def _spw(y: np.ndarray) -> float:
            pos = float(np.sum(y == 1))
            neg = float(np.sum(y == 0))
            return float(max(1.0, (neg / max(1.0, pos))))

        common_kwargs = dict(
            objective="binary:logistic",
            tree_method="hist",
            device=_resolve_device(),
            random_state=int(seed),
            n_jobs=(int(n_jobs) if int(n_jobs) > 0 else -1),
            learning_rate=float(cfg["learning_rate"]),
            max_depth=int(cfg["max_depth"]),
            min_child_weight=float(cfg["min_child_weight"]),
            subsample=float(cfg["subsample"]),
            colsample_bytree=float(cfg["colsample_bytree"]),
            reg_alpha=float(cfg["reg_alpha"]),
            reg_lambda=float(cfg["reg_lambda"]),
            n_estimators=int(cfg["n_estimators"]),
            eval_metric="aucpr",
        )

        # Optional naive random oversampling of positive class
        def _oversample(Xi: np.ndarray, yi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            r = float(oversample_pos)
            if r <= 0.0:
                return Xi, yi
            pos_idx = np.flatnonzero(yi == 1)
            if pos_idx.size == 0:
                return Xi, yi
            rep = int(max(1, np.floor(r)))
            extra = np.repeat(pos_idx, rep)
            X_o = np.concatenate([Xi, Xi[extra]], axis=0)
            y_o = np.concatenate([yi, yi[extra]], axis=0)
            return X_o, y_o

        X_up_tr_np, y_up_tr_np = _oversample(X_tr.values, y_up_tr)
        X_dn_tr_np, y_dn_tr_np = _oversample(X_tr.values, y_dn_tr)

        up_clf = xgb.XGBClassifier(**{**common_kwargs, **{"scale_pos_weight": _spw(y_up_tr_np)}})
        dn_clf = xgb.XGBClassifier(**{**common_kwargs, **{"scale_pos_weight": _spw(y_dn_tr_np)}})
        try:
            up_clf.fit(X_up_tr_np, y_up_tr_np, eval_set=[(X_val.values, y_up_val)], verbose=False)
            dn_clf.fit(X_dn_tr_np, y_dn_tr_np, eval_set=[(X_val.values, y_dn_val)], verbose=False)
        except Exception as e:
            typer.echo(f"Trial fit failed: {e}")
            return float('-inf')

        p_up = up_clf.predict_proba(X_val.values)[:, 1]
        p_dn = dn_clf.predict_proba(X_val.values)[:, 1]
        try:
            auprc_up = float(average_precision_score(y_up_val, p_up))
            auprc_dn = float(average_precision_score(y_dn_val, p_dn))
        except Exception:
            auprc_up = auprc_dn = float('nan')
        val = float(np.nanmean([auprc_up, auprc_dn]))

        # Persist best
        model_up_path = str(tune_dir / f"xgb_impulse_up_trial{trial.number}.json")
        model_dn_path = str(tune_dir / f"xgb_impulse_down_trial{trial.number}.json")
        try:
            if val > float(best_val):
                up_clf.save_model(model_up_path)
                dn_clf.save_model(model_dn_path)
                feat_cols_path = str(Path(model_up_path).with_suffix("").as_posix()) + "_feature_columns.json"
                with open(feat_cols_path, "w") as f:
                    json.dump(list(X_tr.columns), f)
        except Exception:
            pass

        # Log row
        row = {
            "trial": trial.number,
            "value": float(val),
            "auprc_up": float(auprc_up),
            "auprc_down": float(auprc_dn),
            "pos_rate_up": float(np.mean(y_up_val)) if len(y_up_val) else float('nan'),
            "pos_rate_down": float(np.mean(y_dn_val)) if len(y_dn_val) else float('nan'),
            "features_used": ",".join(list(cfg.get("feature_whitelist", []))) if cfg.get("feature_whitelist") else "",
            "extra_timeframes": ",".join(etf) if etf else "",
            "learning_rate": float(cfg["learning_rate"]),
            "max_depth": int(cfg["max_depth"]),
            "min_child_weight": float(cfg["min_child_weight"]),
            "subsample": float(cfg["subsample"]),
            "colsample_bytree": float(cfg["colsample_bytree"]),
            "reg_alpha": float(cfg["reg_alpha"]),
            "reg_lambda": float(cfg["reg_lambda"]),
            "n_estimators": int(cfg["n_estimators"]),
            "thr_up_bps": float(thr_up_bps),
            "thr_dn_bps": float(thr_dn_bps),
            "device": _resolve_device(),
            "model_up": model_up_path,
            "model_down": model_dn_path,
        }
        with open(results_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        return float(val)

    if isinstance(smp, GridSampler):
        try:
            grid_dict = smp.search_space()  # type: ignore[attr-defined]
            grid_size = 1
            for v in grid_dict.values():
                grid_size *= max(1, len(v))
            study.optimize(objective, n_trials=grid_size)
        except Exception:
            study.optimize(objective, n_trials=n_trials)
    else:
        study.optimize(objective, n_trials=n_trials)

    with open(tune_dir / "best_value.txt", "w") as f:
        f.write(str(study.best_value))
    with open(tune_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    typer.echo(f"Best value: {study.best_value}\nBest params: {json.dumps(study.best_params)}")

@app.command()
def xgb_combo_infer(
    reg_model: str = typer.Option(..., help="Path to trained regressor .json"),
    up_model: str = typer.Option(..., help="Path to trained impulse-up classifier .json"),
    down_model: str = typer.Option("", help="Optional impulse-down classifier .json"),
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    exchange: str = typer.Option("bybit"),
    timerange: str = typer.Option("", help="Eval slice YYYYMMDD-YYYYMMDD"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option(""),
    horizon: int = typer.Option(8),
    mu_thr: float = typer.Option(0.0005, help="Threshold on regressed mu (logret)"),
    p_up_thr: float = typer.Option(0.6, help="Impulse-up prob threshold"),
    base_size: float = typer.Option(0.5, help="Normal long size when mu>thr"),
    boost_size: float = typer.Option(1.0, help="Boosted long size when impulse_up"),
    xgb_device: str = typer.Option("auto", help="auto|cpu|cuda"),
    out_csv: str = typer.Option("", help="Optional output CSV path"),
):
    """Combine regressor mu and impulse-up classifier to produce position sizing."""
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing deps. Install xgboost, numpy, pandas.") from e

    # Load dataset
    data_path = _find_data_file_internal(userdir, pair, timeframe, prefer_exchange=exchange)
    if not data_path:
        raise FileNotFoundError("No dataset found.")
    raw = _load_ohlcv_internal(data_path)

    # Load feature columns from models (union)
    def _load_feat_cols(model_path: str) -> list[str]:
        p = Path(model_path)
        cand = p.with_suffix("").as_posix() + "_feature_columns.json"
        if os.path.exists(cand):
            try:
                with open(cand, "r") as f:
                    cols = json.load(f)
                if isinstance(cols, list):
                    return [str(c) for c in cols]
            except Exception:
                return []
        return []

    reg_cols = _load_feat_cols(reg_model)
    up_cols = _load_feat_cols(up_model)
    dn_cols = _load_feat_cols(down_model) if down_model else []
    union_cols = list(dict.fromkeys([*reg_cols, *up_cols, *dn_cols])) if (reg_cols or up_cols or dn_cols) else None

    # Build features once with union
    from rl_lib.features import make_features as _make_features
    etf_list = [s.strip() for s in extra_timeframes.split(",") if s.strip()] if extra_timeframes else None
    feats_all = _make_features(raw, feature_columns=union_cols, mode=feature_mode, basic_lookback=basic_lookback, extra_timeframes=etf_list)

    # Align to valid length for horizon
    H = int(max(1, horizon))
    valid_len = len(feats_all) - H
    feats = feats_all.iloc[:valid_len, :].copy()
    idx = feats.index

    # Prepare separate views for each model (reorder columns)
    def _view(df: pd.DataFrame, cols: list[str] | None) -> pd.DataFrame:
        if not cols:
            return df
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = 0.0
        return out.reindex(columns=cols)

    X_reg = _view(feats, reg_cols) if reg_cols else feats
    X_up = _view(feats, up_cols) if up_cols else feats
    X_dn = _view(feats, dn_cols) if dn_cols else feats

    # Load models
    dev_opt = str(xgb_device).strip().lower()
    dev = ("cuda" if dev_opt == "auto" else dev_opt) if dev_opt in ("cpu", "cuda", "auto") else "cpu"
    reg = xgb.XGBRegressor(device=dev)
    reg.load_model(reg_model)
    upc = xgb.XGBClassifier(device=dev)
    upc.load_model(up_model)
    dnc = None
    if down_model:
        dnc = xgb.XGBClassifier(device=dev)
        dnc.load_model(down_model)

    # Predict
    mu = reg.predict(X_reg.values)
    p_up = upc.predict_proba(X_up.values)[:, 1]
    p_dn = dnc.predict_proba(X_dn.values)[:, 1] if dnc is not None else np.zeros_like(p_up)

    # Combine
    size = np.zeros_like(mu, dtype=float)
    size[p_up >= float(p_up_thr)] = float(boost_size)
    mask_norm = (size == 0.0) & (mu >= float(mu_thr))
    size[mask_norm] = float(base_size)

    out = feats.copy()
    out["mu_pred"] = mu
    out["impulse_up_prob"] = p_up
    out["impulse_down_prob"] = p_dn
    out["long_size"] = size

    # Optional save
    if out_csv:
        Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=True)
        typer.echo(f"Saved combined inference to: {out_csv}")
    else:
        # Print brief summary
        q = np.quantile(mu, [0.9, 0.95, 0.99]) if len(mu) > 100 else [float('nan')]*3
        typer.echo(f"rows={len(out)} mu_thr={mu_thr} p_up_thr={p_up_thr} top_mu={q}")

@app.command()
def xgb_eval(
    model_path: str = typer.Option(..., help="Path to trained regressor .json"),
    pairs: str = typer.Option("BTC/USDT", help="Comma-separated pairs, e.g., 'BTC/USDT,ETH/USDT'"),
    timeframes: str = typer.Option("1h", help="Comma-separated TFs, e.g., '1h,4h,1d'"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    exchange: str = typer.Option("bybit"),
    timerange: str = typer.Option("", help="Eval slice YYYYMMDD-YYYYMMDD"),
    feature_mode: str = typer.Option("full"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("", help="Optional HTFs like '4H,1D' to recompute"),
    horizon: int = typer.Option(1, help="Forward return horizon used during training"),
    ic_metric: str = typer.Option("spearman", help="spearman|pearson"),
    xgb_device: str = typer.Option("auto", help="auto|cpu|cuda"),
    out_csv: str = typer.Option("", help="Optional results CSV path"),
):
    """Evaluate a trained XGB regressor across multiple pairs and timeframes.

    Computes IC (Spearman/Pearson), MSE, direction accuracy for each (pair, timeframe).
    Uses saved feature_columns.json from model to build matching feature layouts.
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing deps. Install xgboost, numpy, pandas.") from e

    pairs_l = [p.strip() for p in pairs.split(',') if p.strip()]
    tfs_l = [t.strip() for t in timeframes.split(',') if t.strip()]
    etf_list = [s.strip() for s in extra_timeframes.split(',') if s.strip()] if extra_timeframes else None

    # Load model and its feature columns
    def _load_feat_cols(mp: str) -> list[str] | None:
        path = Path(mp)
        cand = path.with_suffix("").as_posix() + "_feature_columns.json"
        if os.path.exists(cand):
            try:
                with open(cand, "r") as f:
                    cols = json.load(f)
                if isinstance(cols, list):
                    return [str(c) for c in cols]
            except Exception:
                return None
        return None

    model_cols = _load_feat_cols(model_path)
    dev_opt = str(xgb_device).strip().lower()
    dev = ("cuda" if dev_opt == "auto" else dev_opt) if dev_opt in ("cpu", "cuda", "auto") else "cpu"
    reg = xgb.XGBRegressor(device=dev)
    reg.load_model(model_path)

    # IC function
    def _ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        s1 = pd.Series(y_true)
        s2 = pd.Series(y_pred)
        meth = "spearman" if str(ic_metric).lower() == "spearman" else "pearson"
        v = float(s1.corr(s2, method=meth))
        return v if v == v else float("nan")

    # Compute
    rows = []
    for pr in pairs_l:
        for tf in tfs_l:
            try:
                data_path = _find_data_file_internal(userdir, pr, tf, prefer_exchange=exchange)
                if not data_path:
                    typer.echo(f"Skip: no data for {pr} {tf}")
                    continue
                raw = _load_ohlcv_internal(data_path)
                # Slice
                if timerange:
                    try:
                        s, e = timerange.split('-', 1)
                        start = pd.to_datetime(s) if s else None
                        end = pd.to_datetime(e) if e else None
                        if isinstance(raw.index, pd.DatetimeIndex):
                            idx = raw.index
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

                # Features using model columns for exact layout
                from rl_lib.features import make_features as _make_features
                feats = _make_features(raw, feature_columns=model_cols, mode=feature_mode, basic_lookback=basic_lookback, extra_timeframes=etf_list)

                # Build target and align
                H = int(max(1, horizon))
                c = feats["close"].astype(float).values if "close" in feats.columns else raw["close"].astype(float).values
                logp = pd.Series(np.log(c + 1e-12), index=feats.index)
                y = (logp.shift(-H) - logp).to_numpy()
                valid_len = len(feats) - H
                if valid_len <= 50:
                    typer.echo(f"Skip: insufficient length for {pr} {tf}")
                    continue
                X = feats.iloc[:valid_len, :].copy().values
                y = y[:valid_len]

                # Predict and score
                y_pred = reg.predict(X)
                ic = _ic(y, y_pred)
                mse = float(np.mean((y - y_pred) ** 2))
                acc = float(np.mean(np.sign(y) == np.sign(y_pred)))

                rows.append({
                    "pair": pr,
                    "timeframe": tf,
                    "rows": int(valid_len),
                    "ic": float(ic),
                    "mse": float(mse),
                    "acc_dir": float(acc),
                })
            except Exception as e:
                typer.echo(f"Eval failed for {pr} {tf}: {e}")
                continue

    # Output
    if out_csv:
        Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with open(out_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=["pair","timeframe","rows","ic","mse","acc_dir"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        typer.echo(f"Saved xgb_eval results to: {out_csv}")
    # Print results concise
    for r in rows:
        typer.echo(f"{r['pair']} {r['timeframe']} rows={r['rows']} IC={r['ic']:.5f} MSE={r['mse']:.6f} ACC={r['acc_dir']:.3f}")

@app.command()
def forecast_eval(
    model_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "forecaster.pt"), "--model-path", "--model_path"),
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    exchange: str = typer.Option("bybit", help="Prefer this exchange's dataset when multiple exist"),
    timerange: str = typer.Option("", help="YYYYMMDD-YYYYMMDD for evaluation slice"),
    feature_mode: str = typer.Option("full", help="full|basic|ohlcv"),
    basic_lookback: int = typer.Option(64),
    extra_timeframes: str = typer.Option("", help="e.g., '4H,1D'"),
    device: str = typer.Option("cuda"),
    max_windows: int = typer.Option(1200),
    outdir: str = typer.Option("", help="Output dir for plots/CSVs; default under model folder"),
    make_animation: bool = typer.Option(False, help="Save animation"),
    anim_fps: int = typer.Option(12, help="Animation FPS"),
    animation_mode: str = typer.Option("next", help="next|path"),
):
    """Evaluate a trained forecaster and save plots and CSVs for inspection."""
    etf_list = [s.strip() for s in extra_timeframes.split(",") if s.strip()] if extra_timeframes else []
    report = evaluate_forecaster(
        model_path=model_path,
        userdir=userdir,
        pair=pair,
        timeframe=timeframe,
        feature_mode=feature_mode,
        basic_lookback=basic_lookback,
        extra_timeframes=(etf_list or None),
        timerange=(timerange or None),
        device=device,
        max_windows=int(max_windows),
        outdir=(outdir or None),
        make_animation=bool(make_animation),
        anim_fps=int(anim_fps),
        animation_mode=str(animation_mode),
        prefer_exchange=exchange,
    )
    typer.echo(json.dumps(report, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))


if __name__ == "__main__":
    app()

