#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACK_DIR="$ROOT_DIR/backtest_server"
FRONT_DIR="$ROOT_DIR/backtest_terminal"

echo "[+] Repo root: $ROOT_DIR"

# Start backend (FastAPI on 8501)
echo "[+] Starting backend (FastAPI) on :8080"
(
  cd "$BACK_DIR"
  # Create venv if missing
  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
  # Minimal deps for inference server
  pip install -q fastapi uvicorn[standard] pydantic pandas numpy pyarrow stable-baselines3 sb3-contrib gymnasium cloudpickle >/dev/null
  # Prefer CPU by default to avoid CUDA issues on Mac unless RL_DEVICE is set
  export RL_DEVICE="${RL_DEVICE:-cpu}"
  # Run server
  uvicorn backtest_server.main:app --host 0.0.0.0 --port 8080 --reload &
  echo $! > "$BACK_DIR/.uvicorn.pid"
  deactivate || true
) 

# Start frontend (Next.js on 3000)
echo "[+] Starting frontend (Next.js) on :8501"
(
  cd "$FRONT_DIR"
  npm install --silent >/dev/null 2>&1 || true
  # Next: dev server default is 3000; bind to 8501
  PORT=8501 npm run dev --silent &
  echo $! > "$FRONT_DIR/.next.pid"
)

BACK_PID="$(cat "$BACK_DIR/.uvicorn.pid" 2>/dev/null || echo "")"
FRONT_PID="$(cat "$FRONT_DIR/.next.pid" 2>/dev/null || echo "")"

cleanup() {
  echo "\n[+] Shutting down..."
  if [ -n "$FRONT_PID" ] && ps -p "$FRONT_PID" >/dev/null 2>&1; then
    kill "$FRONT_PID" 2>/dev/null || true
  fi
  if [ -n "$BACK_PID" ] && ps -p "$BACK_PID" >/dev/null 2>&1; then
    kill "$BACK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[+] Backend:  http://127.0.0.1:8080/health"
echo "[+] Frontend: http://127.0.0.1:8501"

# macOS: open browser tab (optional)
if command -v open >/dev/null 2>&1; then
  (sleep 1 && open "http://127.0.0.1:8501") >/dev/null 2>&1 || true
fi

echo "[i] Press Ctrl+C to stop both servers."
wait


