# SnakeLR

Gymnasium-compatible Snake environment with image observations, optional pygame rendering, web UI, and RLlib training script.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

- Train with RLlib PPO:
```bash
python scripts/train_rllib.py --height 15 --width 15 --steps 200
```

- Manual play (pygame):
```bash
python scripts/play_pygame.py
```

- Web UI (FastAPI):
```bash
uvicorn web.app:app --reload
# open http://127.0.0.1:8000
```

## Docker

```bash
docker compose up --build
# open http://localhost:8000
```

## RLlib registration

Env id: `Snake-v0`. Use `snakelr.register.register_env()` or register via Tune registry as shown in `scripts/train_rllib.py`.

## Observation / Action

- Observation: HxWx3 RGB NumPy array of board.
- Action space: Discrete(4) — 0 up, 1 down, 2 left, 3 right.

## Rewards

- +1 food, -1 death, -0.01 per step.

## Optional Features

Enable via `SnakeConfig` or `env_config`:
- Poison (`enable_poison`)
- Obstacles (`enable_obstacles`)
- Multi-food (`enable_multi_food`)
