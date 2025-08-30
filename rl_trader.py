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

from rl_lib.train_sb3 import TrainParams, train_ppo_from_freqtrade_data, validate_trained_model


app = typer.Typer(add_completion=False)


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


@app.command()
def train(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    window: int = typer.Option(128),
    timesteps: int = typer.Option(200_000),
    model_out: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "rl_ppo.zip")),
    arch: str = typer.Option("mlp", help="mlp|lstm|transformer|transformer_big|transformer_hybrid"),
    fee_bps: float = typer.Option(6.0, help="Trading fee in basis points (e.g., 6.0 = 0.06%)"),
    slippage_bps: float = typer.Option(2.0, help="Slippage in basis points"),
    idle_penalty_bps: float = typer.Option(0.02, help="Idle penalty in bps when flat (applied in env)"),
    exchange: str = typer.Option("bybit", help="Prefer this exchange's dataset when multiple exist"),
    device: str = typer.Option("cuda", help="Device for training: cuda|cpu"),
    # New options
    seed: int = typer.Option(42, help="Random seed"),
    reward_type: str = typer.Option("raw", help="raw|vol_scaled|sharpe_proxy"),
    vol_lookback: int = typer.Option(20, help="Volatility lookback for reward shaping"),
    turnover_penalty_bps: float = typer.Option(0.0, help="Extra turnover penalty in bps per change"),
    dd_penalty: float = typer.Option(0.0, help="Coefficient for drawdown penalty in sharpe_proxy"),
    min_hold_bars: int = typer.Option(0, help="Minimum bars to hold before closing/flip"),
    cooldown_bars: int = typer.Option(0, help="Cooldown bars after closing before re-entry"),
    random_reset: bool = typer.Option(False, help="Randomize episode start index"),
    episode_max_steps: int = typer.Option(0, help="Max steps per episode (0 = run to dataset end)"),
    feature_mode: str = typer.Option("full", help="full|basic (basic: close_z, change, d_hl)"),
    basic_lookback: int = typer.Option(64, help="Lookback for basic close_z standardization"),
    # Eval options
    eval_freq: int = typer.Option(100000, help="Evaluate every N steps (0 disables)"),
    n_eval_episodes: int = typer.Option(3, help="Episodes per eval"),
    eval_max_steps: int = typer.Option(2000, help="Max steps per eval rollout"),
):
    """Train PPO on downloaded data using Stable-Baselines3."""
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
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        eval_max_steps=eval_max_steps,
    )
    out = train_ppo_from_freqtrade_data(params)
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
def validate(
    pair: str = typer.Option("BTC/USDT"),
    timeframe: str = typer.Option("1h"),
    userdir: str = typer.Option(str(Path(__file__).resolve().parent / "freqtrade_userdir")),
    window: int = typer.Option(128),
    model_path: str = typer.Option(str(Path(__file__).resolve().parent / "models" / "rl_ppo.zip")),
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
    min_hold_bars: int = typer.Option(0, help="Minimum bars to hold before closing/flip"),
    cooldown_bars: int = typer.Option(0, help="Cooldown bars after closing before re-entry"),
    random_reset: bool = typer.Option(False, help="Randomize episode start index"),
    episode_max_steps: int = typer.Option(0, help="Max steps per episode (0 = run to dataset end)"),
    feature_mode: str = typer.Option("full", help="full|basic (basic: close_z, change, d_hl)"),
    basic_lookback: int = typer.Option(64, help="Lookback for basic close_z standardization"),
):
    """Run a quick validation rollout on eval split and print summary (actions, entries, equity)."""
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
    )
    os.environ["RL_DEVICE"] = device
    _ = validate_trained_model(params, max_steps=max_steps, deterministic=deterministic, timerange=timerange)


if __name__ == "__main__":
    app()

