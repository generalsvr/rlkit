Run FastAPI server

Local dev:

```bash
cd backtest_server
python3 -m venv .venv
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

When using the run script from repo root:

```bash
./run_backtest_terminal.sh
```

Note: Use `main:app` as the ASGI target when the CWD is `backtest_server/`.

