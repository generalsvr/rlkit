from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from .envs.snake_env import SnakeEnv, SnakeConfig


ENV_ID = "Snake-v0"


def make_env(config: Optional[Dict] = None, render_mode: Optional[str] = None) -> SnakeEnv:
    return SnakeEnv(config=config, render_mode=render_mode)


def register_env():
    # Gymnasium registration
    try:
        register(
            id=ENV_ID,
            entry_point="snakelr.envs.snake_env:SnakeEnv",
        )
    except Exception:
        # ignore if already registered
        pass

    # RLlib registration
    try:
        import ray
        from ray.tune.registry import register_env as rllib_register_env

        def env_creator(env_config: Dict):
            render_mode = env_config.pop("render_mode", None)
            return SnakeEnv(config=env_config, render_mode=render_mode)

        rllib_register_env(ENV_ID, env_creator)
    except Exception:
        # RLlib may not be installed in minimal environments
        pass
