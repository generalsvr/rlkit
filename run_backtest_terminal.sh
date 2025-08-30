#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACK_DIR="$ROOT_DIR/backtest_server"
FRONT_DIR="$ROOT_DIR/backtest_terminal"

echo "[+] Repo root: $ROOT_DIR"

# Ensure npm/node are available; try OS package manager first, else install NVM locally
ensure_npm() {
  if command -v npm >/dev/null 2>&1 && command -v node >/dev/null 2>&1; then
    return 0
  fi
  echo "[i] npm/node not found; attempting installation..."
  if command -v apt-get >/dev/null 2>&1 && [ "${EUID:-$(id -u)}" -eq 0 ]; then
    echo "[i] Installing Node.js via NodeSource (Debian/Ubuntu)..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y >/dev/null 2>&1 || true
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null 2>&1 || true
    apt-get install -y nodejs >/dev/null 2>&1 || apt-get install -y nodejs npm >/dev/null 2>&1 || true
  elif command -v yum >/dev/null 2>&1 && [ "${EUID:-$(id -u)}" -eq 0 ]; then
    echo "[i] Installing Node.js via NodeSource (RHEL/CentOS)..."
    curl -fsSL https://rpm.nodesource.com/setup_20.x | bash - >/dev/null 2>&1 || true
    yum install -y nodejs >/dev/null 2>&1 || true
  fi
  if command -v npm >/dev/null 2>&1; then
    return 0
  fi
  # Fallback: install NVM under current user (no root required)
  echo "[i] Falling back to NVM install (user-level)..."
  export NVM_DIR="$HOME/.nvm"
  mkdir -p "$NVM_DIR"
  if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash >/dev/null 2>&1 || true
  fi
  # shellcheck disable=SC1090
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  if command -v nvm >/dev/null 2>&1; then
    nvm install --lts >/dev/null 2>&1 || nvm install 20 >/dev/null 2>&1 || true
    nvm use --lts >/dev/null 2>&1 || nvm use 20 >/dev/null 2>&1 || true
  fi
}

echo "[+] Starting frontend (Next.js) on :8501"
(
  cd "$FRONT_DIR"
  ensure_npm
  # If nvm is installed, load it to expose npm/node in this subshell
  if [ -s "$HOME/.nvm/nvm.sh" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.nvm/nvm.sh"
    nvm use --lts >/dev/null 2>&1 || true
  fi
  npm install --silent >/dev/null 2>&1 || true
  PORT=8501 npm run dev --silent &
  echo $! > "$FRONT_DIR/.next.pid"
)

echo "[+] Starting backend (FastAPI) on :8080 (this may take a minute on first run)"
(
  cd "$BACK_DIR"
  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null 2>&1 || true
  # Minimal deps for inference server
  pip install -q fastapi uvicorn[standard] pydantic pandas numpy pyarrow stable-baselines3 sb3-contrib gymnasium cloudpickle >/dev/null 2>&1 || true
  export RL_DEVICE="${RL_DEVICE:-cpu}"
  uvicorn backtest_server.main:app --host 0.0.0.0 --port 8080 --reload &
  echo $! > "$BACK_DIR/.uvicorn.pid"
  deactivate || true
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


