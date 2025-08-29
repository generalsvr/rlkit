## RL Trading Pipeline (Freqtrade + Stable-Baselines3)

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install freqtrade
```

### Docker (official Freqtrade RL image)
```bash
# Pull image
docker pull freqtradeorg/freqtrade:2025.7_freqairl

# Prepare userdir mapping
mkdir -p freqtrade_userdir/models
cp -R rl_lib freqtrade_userdir/

# Download data to userdir
docker run --rm -it \
  -v "$PWD/freqtrade_userdir:/freqtrade/user_data" \
  freqtradeorg/freqtrade:2025.7_freqairl \
  download-data --userdir /freqtrade/user_data \
  --exchange binance --pairs BTC/USDT --timeframes 1h --timerange 20140101-

# Train locally (SB3) and save model to models/rl_ppo.zip
python rl_trader.py train --pair BTC/USDT --timeframe 1h --window 128 --timesteps 200000 --arch mlp
cp models/rl_ppo.zip freqtrade_userdir/models/

# Backtest using the model inside the official container
docker run --rm -it \
  -v "$PWD/freqtrade_userdir:/freqtrade/user_data" \
  -e RL_MODEL_PATH=/freqtrade/user_data/models/rl_ppo.zip \
  freqtradeorg/freqtrade:2025.7_freqairl \
  backtesting --userdir /freqtrade/user_data \
  --config /freqtrade/user_data/config.json \
  --strategy RLStrategy --timeframe 1h --pairs BTC/USDT \
  --timerange 20240101-20250101
```

### 1) Download data with Freqtrade
```bash
python rl_trader.py download --pair BTC/USDT --timeframe 1h --timerange 20190101-20240101 --exchange binance
```

### 2) Train PPO on downloaded data
```bash
python rl_trader.py train --pair BTC/USDT --timeframe 1h --window 128 --timesteps 200000
```
Model saved to `models/rl_ppo.zip`.

### 3) Backtest in Freqtrade using RLStrategy
```bash
python rl_trader.py backtest --pair BTC/USDT --timeframe 1h --timerange 20220101-20240101
```

Notes:
- Strategy: `freqtrade_userdir/strategies/RLStrategy.py`
- Config example: `freqtrade_userdir/config.example.json`
- Set `RL_MODEL_PATH` env var to override default model path.


# Download and install TA-Lib C library

apt update && apt install build-essential wget && cd /tmp && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib/ && ./configure --prefix=/usr && make && make install && ldconfig