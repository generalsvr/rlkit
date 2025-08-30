Backtest Agent Terminal (Next.js + FastAPI)

Getting Started

1) Backend (FastAPI on 8080)

```bash
cd ../backtest_server
python3 -m venv .venv
source .venv/bin/activate
python main.py
```

Endpoints:
- POST /candles { pair, timeframe, timerange? }
- POST /run { pair, timeframe, model_path, timerange?, window, reward_type, ... }

2) Frontend (Next.js on 8501)

```bash
cd ../backtest_terminal
npm install
npm run dev  # serves on 8501
```

Use the UI to:
- Load candles (full, scrollable)
- Run agent from arbitrary timerange
- Visualize actions as markers and inspect logs
