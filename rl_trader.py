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


app = typer.Typer(add_completion=False)


def _parse_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


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
    """Ensure dataset exists; if not, download via Freqtrade non-interactively."""
    from rl_lib.train_sb3 import _find_data_file as _find
    hit = _find(userdir, pair, timeframe, prefer_exchange=exchange)
    if hit and os.path.exists(hit):
        return hit
    Path(userdir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "freqtrade", "download-data",
        "--pairs", pair,
        "--timeframes", timeframe,
        "--userdir", userdir,
        "--timerange", timerange,
        "--exchange", exchange,
        "--data-format-ohlcv", fmt,
    ]
    typer.echo(f"Downloading missing dataset: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # Try find again
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
                        # Try common fields
                        def _get(*keys):
                            for k in keys:
                                if isinstance(data, dict) and k in data and isinstance(data[k], (int, float)):
                                    return data[k]
                            return None
                        # Flat keys
                        out['bt_total_profit_abs'] = _get('profit_total_abs', 'total_profit', 'net_profit')
                        out['bt_total_profit_pct'] = _get('profit_total_pct', 'total_profit_pct')
                        out['bt_total_trades'] = _get('total_trades', 'trades')
                        out['bt_win_rate'] = _get('winrate', 'win_rate')
                        out['bt_profit_factor'] = _get('profit_factor', 'pf')
                        # Stop at first JSON that yields any metric
                        if any(v is not None for v in out.values()):
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
    timerange: str = typer.Option("20220101-20240101"),
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
    feature_mode: str = typer.Option("basic"),
    basic_lookback: int = typer.Option(128),
    windows_list: str = typer.Option("", help="Comma-separated window sizes; if empty, uses --window"),
    min_hold_bars_list: str = typer.Option("", help="Comma-separated min-hold bars values"),
    cooldown_bars_list: str = typer.Option("", help="Comma-separated cooldown bars values"),
    # Eval and early stop
    eval_max_steps: int = typer.Option(5000),
    eval_freq: int = typer.Option(50000, help="Evaluate every N steps; enable >0 for early stopping"),
    early_stop_metric: str = typer.Option("sharpe", help="sharpe|final_equity|max_drawdown"),
    early_stop_patience: int = typer.Option(3),
    early_stop_min_delta: float = typer.Option(0.0),
    early_stop_degrade_ratio: float = typer.Option(0.0),
    # Auto backtest
    auto_backtest: bool = typer.Option(True, help="Run freqtrade backtest per model"),
    backtest_timerange: str = typer.Option("", help="YYYYMMDD-YYYYMMDD for freqtrade backtests; empty uses training eval slice"),
    backtest_exchange: str = typer.Option("bybit"),
    outdir: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "sweeps")),
    max_trials: int = typer.Option(50, help="Hard cap to avoid huge grids"),
):
    """Run a small hyperparameter grid sequentially and log results to CSV."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(outdir) / ts
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results_csv = sweep_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_path","window","min_hold_bars","cooldown_bars","reward_type","ent_coef","learning_rate","n_steps","batch_size","fee_bps","slippage_bps","seed",
            "final_equity","sharpe","max_drawdown","time_in_position_frac",
            "bt_total_profit_pct","bt_total_profit_abs","bt_total_trades","bt_win_rate","bt_profit_factor",
            "bt_zip","bt_meta","bt_run_id"
        ])
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

    combos = list(product(rt_list, ec_list, lr_list, ns_list, bs_list, fee_vals, slip_vals, seed_vals, win_vals, mh_vals, cd_vals))
    if len(combos) > max_trials:
        combos = combos[:max_trials]

    typer.echo(f"Running {len(combos)} trials. Results -> {results_csv}")

    for idx, (rt, ec, lr, ns, bs, fee, slip, sd, wv, mhv, cdv) in enumerate(combos, start=1):
        tag = f"{arch}_win-{wv}_mh-{mhv}_cd-{cdv}_rt-{rt}_ec-{ec}_lr-{lr}_ns-{ns}_bs-{bs}_fee-{fee}_slip-{slip}_seed-{sd}"
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
            fee_bps=fee,
            slippage_bps=slip,
            idle_penalty_bps=0.0,
            reward_type=rt,
            vol_lookback=20,
            turnover_penalty_bps=turnover_penalty_bps,
            dd_penalty=0.05 if rt == "sharpe_proxy" else 0.0,
            min_hold_bars=mhv,
            cooldown_bars=cdv,
            random_reset=True,
            episode_max_steps=4096,
            feature_mode=feature_mode,
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
            report = validate_trained_model(params, max_steps=eval_max_steps, deterministic=True)
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
                "window": int(wv),
                "min_hold_bars": int(mhv),
                "cooldown_bars": int(cdv),
                "reward_type": rt,
                "ent_coef": ec,
                "learning_rate": lr,
                "n_steps": ns,
                "batch_size": bs,
                "fee_bps": fee,
                "slippage_bps": slip,
                "seed": sd,
                "final_equity": float(report.get("final_equity", float("nan"))),
                "sharpe": float(report.get("sharpe", float("nan"))),
                "max_drawdown": float(report.get("max_drawdown", float("nan"))),
                "time_in_position_frac": float(report.get("time_in_position_frac", float("nan"))),
                "bt_total_profit_pct": float(bt_metrics.get("bt_total_profit_pct", float("nan"))) if auto_backtest else float("nan"),
                "bt_total_profit_abs": float(bt_metrics.get("bt_total_profit_abs", float("nan"))) if auto_backtest else float("nan"),
                "bt_total_trades": int(bt_metrics.get("bt_total_trades", 0)) if auto_backtest else 0,
                "bt_win_rate": float(bt_metrics.get("bt_win_rate", float("nan"))) if auto_backtest else float("nan"),
                "bt_profit_factor": float(bt_metrics.get("bt_profit_factor", float("nan"))) if auto_backtest else float("nan"),
                "bt_zip": str(bt_metrics.get("bt_zip", "")) if auto_backtest else "",
                "bt_meta": str(bt_metrics.get("bt_meta", "")) if auto_backtest else "",
                "bt_run_id": str(bt_metrics.get("bt_run_id", "")) if auto_backtest else "",
            }
            with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
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


if __name__ == "__main__":
    app()

