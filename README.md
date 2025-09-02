## RL Trading Pipeline (Freqtrade + Stable-Baselines3)

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements.extra.txt
pip install freqtrade

apt update && apt install build-essential wget && cd /tmp && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib/ && ./configure --prefix=/usr && make && make install && ldconfig && cd /workspace && git clone https://github.com/generalsvr/rlkit && cd rlkit && pip install stable-baselines3[extra] && pip install -r requirements.txt
```

### Futures Mode (USDT-M, offline OHLCV)
- Strategy supports long/short via RL (`can_short=True`).
- Backtesting uses local OHLCV parquet. No tickers/orderbooks needed.

Config (save to `freqtrade_userdir/config.json`):
```json
{
  "timeframe": "1h",
  "user_data_dir": "/workspace/rlkit/freqtrade_userdir",
  "strategy": "RLStrategy",
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "dry_run": true,
  "max_open_trades": 1,
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "dataformat_ohlcv": "parquet",
  "entry_pricing": { "price_side": "same", "use_order_book": false },
  "exit_pricing": { "price_side": "same", "use_order_book": false },
  "pairlists": [ { "method": "StaticPairList" } ],
  "exchange": {
    "name": "bybit",
    "key": "",
    "secret": "",
    "pair_whitelist": ["BTC/USDT:USDT"],
    "use_exchange": false
  }
}
```

- Alternative exchange names:
  - Binance USDT-M: set `exchange.name` to `binanceusdm` and `pair_whitelist` to `"BTC/USDT"`. Ensure data path matches exchange folder.

### 1) Download data
Bybit (recommended for futures path):
```bash
freqtrade download-data \
  --exchange bybit \
  --pairs "BTC/USDT:USDT" \
  --timeframes 1h \
  --userdir /workspace/rlkit/freqtrade_userdir \
  --data-format-ohlcv parquet --erase --timerange 20190101- | cat
```

### 2) Train PPO (Hybrid Transformer)
Minimal (MLP):
```bash
python rl_trader.py train --pair "BTC/USDT:USDT" --timeframe 1h \
  --window 128 --timesteps 300000 --arch mlp \
  --fee-bps 6 --slippage-bps 10 --idle-penalty-bps 0.0 \
  --reward-type vol_scaled --vol-lookback 20 \
  --turnover-penalty-bps 2.0 --min-hold-bars 3 --cooldown-bars 1 \
  --random-reset --episode-max-steps 4096 \
  --feature-mode basic --basic-lookback 128 \
  --seed 42 --device cuda \
  --eval-freq 50000 --n-eval-episodes 1 --eval-max-steps 2000
```
Hybrid Transformer (recommended):
```bash
python rl_trader.py train --pair "BTC/USDT:USDT" --timeframe 1h \
  --window 128 --timesteps 2000000 --arch transformer_hybrid \
  --fee-bps 6 --slippage-bps 10 --idle-penalty-bps 0.0 \
  --reward-type vol_scaled --vol-lookback 20 \
  --turnover-penalty-bps 2.0 --min-hold-bars 3 --cooldown-bars 1 \
  --random-reset --episode-max-steps 4096 \
  --feature-mode basic --basic-lookback 128 \
  --seed 42 --device cuda \
  --eval-freq 50000 --n-eval-episodes 1 --eval-max-steps 2000
```
MultiTF:
```bash
python rl_trader.py train-multi   --pairs "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,BNB/USDT:USDT"   --timeframe 1h   --userdir freqtrade_userdir   --window 128   --timesteps 3000000   --model-out ./models/x.zip   --arch transformer_hybrid   --exchange bybit   --device cuda   --seed 42   --reward-type sharpe_proxy   --vol-lookback 20   --fee-bps 6   --slippage-bps 10   --turnover-penalty-bps 2   --min-hold-bars 3   --cooldown-bars 1   --random-reset   --episode-max-steps 4096   --feature-mode basic   --basic-lookback 128   --extra-timeframes "4H,1D"   --eval-freq 49152  --n-eval-episodes 1   --eval-max-steps 2000 --early-stop-metric sharpe --early-stop-patience 3 --early-stop-min-delta 0.001  --early-stop-degrade-ratio 0.2
```

### 2b) Multi-ticker training (auto-download)

Train across multiple symbols with vectorized envs. Automatically ensures datasets exist for 1h,4h,1d,1w (downloads missing ones via Freqtrade).

```bash
python rl_trader.py train_multi \
  --pairs "BTC/USDT:USDT,ETH/USDT:USDT" \
  --timeframe 1h \
  --userdir /workspace/rlkit/freqtrade_userdir \
  --model-out ./models/multi_ppo.zip \
  --exchange bybit \
  --timerange 20190101- \
  --feature-mode basic --extra-timeframes "4H,1D" \
  --arch transformer_hybrid --device cuda --timesteps 800000
```

Notes:
- Auto-download grabs missing datasets for 1h,4h,1d,1w by default. Disable with `--no-autofetch`.
- Use `--pairs` to add more symbols (comma-separated). Ensure the exchange supports them.

### 3) Validate
```bash
python rl_trader.py validate --pair "BTC/USDT:USDT" --timeframe 1h \
  --window 128 --model-path /workspace/rlkit/models/rl_ppo.zip \
  --max-steps 5000 --device cuda --timerange 20250101-
```
- Prints summary including final equity, Sharpe, Sortino, and MaxDD.

### 4) Backtest with RLStrategy
```bash
python rl_trader.py backtest --pair "BTC/USDT:USDT" --timeframe 1h   --window 128 --model-path /workspace/rlkit/models/x.zip  --device cuda --timerange 20250101-
```

### 5) SWEEP

```bash
python rl_trader.py sweep \
  --pair BTC/USDT:USDT \
  --timeframe 1h \
  --userdir /workspace/rlkit/freqtrade_userdir \
  --timesteps 300000 \
  --arch transformer_hybrid \
  --device cuda \
  --reward-types sharpe_proxy \
  --ent-coefs 0.02 \
  --lrs 0.0003 \
  --n-steps-list 2048 \
  --batch-sizes 256 \
  --fee-list 6 \
  --slip-list 2 \
  --seeds 42 \
  --extra-timeframes 4H,1D \
  --turnover-penalty-bps 4 \
  --feature-modes full \
  --basic-lookback 128 \
  --windows-list 128,192,256 \
  --min-hold-bars-list 1,2 \
  --cooldown-bars-list 2,3 \
  --position-penalty-bps-list 1,2,3 \
  --loss-hold-penalty-bps-list 1,2,3 \
  --cvar-alpha-list 0.05 \
  --cvar-coef-list 0.1,0.2,0.3 \
  --max-position-bars-list 64,96,128 \
  --eval-max-steps 4000 \
  --eval-freq 49152 \
  --eval-timerange 20240101-20250101 \
  --early-stop-metric sharpe \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.001 \
  --early-stop-degrade-ratio 0.2 \
  --auto-backtest \
  --backtest-timerange 20250101- \
  --backtest-exchange bybit \
  --autofetch \
  --timerange 20170101- \
  --exchange bybit \
  --outdir /workspace/rlkit/models/sweeps \
  --max-trials 99999
```

### 5b) Optuna Tuning (Bayesian/Random/Grid)

Bayesian (TPE) example:
```bash
python rl_trader.py tune \
  --pair BTC/USDT:USDT --timeframe 1h \
  --userdir /workspace/rlkit/freqtrade_userdir \
  --timesteps 300000 --arch transformer_hybrid --device cuda \
  --sampler tpe --n-trials 30 --seed 42 \
  --eval-max-steps 4000 --eval-freq 49152 \
  --early-stop-metric sharpe --early-stop-patience 3 \
  --auto-backtest --backtest-exchange bybit \
  --outdir /workspace/rlkit/models/optuna
```

Grid search with Optuna's GridSampler:
```bash
python rl_trader.py tune --sampler grid --n-trials 999999 \
  --pair BTC/USDT:USDT --timeframe 1h --userdir /workspace/rlkit/freqtrade_userdir \
  --timesteps 300000 --device cuda --arch transformer_hybrid
```

Outputs:
- Per-trial models, `results.csv`, `best_params.json`, and `best_value.txt` under `models/optuna/<timestamp>/`.

### 6) DECODER OHLCV

```bash
python rl_trader.py forecast_train \
  --pair BTC/USDT --timeframe 1h \
  --feature-mode ohlcv \
  --window 128 --horizon 16 \
  --epochs 20 --device cuda \
  --model-out freqtrade_userdir/models/forecaster.pt
  
```

### 7. LOGRET

```bash
python rl_trader.py forecast-train   --pair BTC/USDT --timeframe 15m   --feature-mode full --extra-timeframes 1h,4h,1D   --target-mode logret --forecast-arch decoder_only   --window 512 --horizon 4   --d-model 128 --nhead 4 --num-encoder-layers 2 --num-decoder-layers 2 --ff-dim 256   --dropout 0.3 --weight-decay 0.003 --learning-rate 1e-4 --batch-size 512 --epochs 10 --device cuda   --autofetch --exchange bybit --download-timerange 20160101-   --model-out freqtrade_userdir/models/forecaster_15m_logret_decoder.pt
```

### Notes
- Strategy file: `freqtrade_userdir/strategies/RLStrategy.py`
- RL signals: `rl_lib/signal.py` produces `enter_long/exit_long/enter_short/exit_short` for Freqtrade.
- Normalization: `VecNormalize` stats saved to `models/vecnormalize.pkl` are reused during inference.
- Features:
  - basic: `close_z`, `change`, `d_hl`
  - full: `logret`, `hl_range`, `upper_wick`, `lower_wick`, `rsi`, `vol_z`, `atr`, `ret_std_14` (MACD/Bollinger removed)
- Costs and realism: fees/slippage in bps are charged, flips double-charged; optional min-hold and cooldown.
- Eval: PnL (final equity) printed every `--eval-freq` steps; cap eval rollout length with `--eval-max-steps`.

### Troubleshooting
- "No data found": rerun download with `--erase` or ensure parquet path matches exchange folder.
- "No pair in whitelist": use correct pair symbol (`BTC/USDT:USDT` for Bybit futures).
- CPU runs: set `--device cpu`.

 
