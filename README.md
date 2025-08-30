## RL Trading Pipeline (Freqtrade + Stable-Baselines3)

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install freqtrade
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

If you already have Binance parquet and want to reuse it for Bybit backtesting:
```bash
mkdir -p /workspace/rlkit/freqtrade_userdir/data/bybit
ln -sf /workspace/rlkit/freqtrade_userdir/data/binance/BTC_USDT-1h.parquet \
      /workspace/rlkit/freqtrade_userdir/data/bybit/BTC_USDT_USDT-1h.parquet
```

### 2) Train PPO
Transformer (standard):
```bash
python rl_trader.py train --pair BTC/USDT --timeframe 1h --window 128 --timesteps 300000 --arch transformer
```
Transformer (bigger):
```bash
python rl_trader.py train --pair BTC/USDT --timeframe 1h --window 128 --timesteps 400000 --arch transformer_big
```

### 3) Backtest with RLStrategy
```bash
export RL_MODEL_PATH=/workspace/rlkit/models/rl_ppo.zip
export RL_WINDOW=128
freqtrade backtesting \
  --userdir /workspace/rlkit/freqtrade_userdir \
  --config /workspace/rlkit/freqtrade_userdir/config.json \
  --strategy RLStrategy \
  --timeframe 1h \
  --pairs "BTC/USDT:USDT" \
  --timerange 20240101-20250101 \
  --data-format-ohlcv parquet | cat
```

### Notes
- Strategy file: `freqtrade_userdir/strategies/RLStrategy.py`
- RL signals: `rl_lib/signal.py` produce `enter_long/exit_long/enter_short/exit_short` which map directly to Freqtrade’s unified API.
- Normalization: `VecNormalize` stats saved to `models/vecnormalize.pkl` are used during inference.
- Troubleshooting:
  - "Ticker pricing not available": ensure `use_exchange=false` and `price_side="same"`.
  - "No pair in whitelist": use correct pair symbol (`BTC/USDT:USDT` for Bybit futures), and ensure parquet exists under matching exchange folder.
  - "No data found": re-run download with `--erase` or symlink data to the correct path.

# Download and install TA-Lib C library

git clone https://github.com/generalsvr/rlkit && cd rlkit && apt update && apt install build-essential wget && cd /tmp && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib/ && ./configure --prefix=/usr && 
make && make install && ldconfig && cd /workspace/rlkit && pip install -r requirements.txt 