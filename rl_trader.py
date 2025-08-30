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
    arch: str = typer.Option("mlp", help="mlp|lstm|transformer"),
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
):
    """Backtest trained RL model inside Freqtrade."""
    env = os.environ.copy()
    env["RL_MODEL_PATH"] = model_path
    # Ensure minimal config exists
    config_path = Path(userdir) / "config.json"
    if not config_path.exists():
        cfg = {
            "timeframe": timeframe,
            "user_data_dir": str(userdir),
            "strategy": "RLStrategy",
            "exchange": {
                "name": "binance",
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
):
    """Run a quick validation rollout on eval split and print summary (actions, entries, equity)."""
    params = TrainParams(
        userdir=userdir,
        pair=pair,
        timeframe=timeframe,
        window=window,
        model_out_path=model_path,
    )
    _ = validate_trained_model(params, max_steps=max_steps, deterministic=deterministic)


if __name__ == "__main__":
    app()

