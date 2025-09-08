from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from snakelr.envs.snake_env import SnakeEnv, SnakeConfig
from snakelr.register import ENV_ID
# Ensure custom model is registered
from snakelr.models.cnn_model import SimpleCNNGAP  # noqa: F401


def env_creator(env_config):
    return SnakeEnv(config=env_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--framework", type=str, default="torch", choices=["torch", "tf", "tf2"])
    args = parser.parse_args()

    register_env(ENV_ID, lambda cfg: env_creator(cfg))

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(
            ENV_ID,
            env_config={
                "board_size": (args.height, args.width),
                "enable_multi_food": True,
            },
        )
        .framework(args.framework)
        .env_runners(num_env_runners=args.num_workers)
        .resources(num_gpus=0)
        .training(
            model={
                "custom_model": "simple_cnn_gap",
            }
        )
    )

    stop = {"training_iteration": args.steps}

    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=stop,
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=5,
                num_to_keep=3,
                checkpoint_at_end=True,
            ),
        ),
        param_space=config.to_dict(),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
