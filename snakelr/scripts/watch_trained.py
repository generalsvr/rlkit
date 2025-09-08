from __future__ import annotations

import argparse
import time

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from snakelr.envs.snake_env import SnakeEnv
from snakelr.register import ENV_ID
# Ensure custom model is registered
from snakelr.models.cnn_model import SimpleCNNGAP  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to RLlib checkpoint directory (checkpoint_000xxx)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=15)
    args = parser.parse_args()

    # Register env id used during training
    register_env(ENV_ID, lambda cfg: SnakeEnv(config=cfg))

    algo = Algorithm.from_checkpoint(args.checkpoint)

    env = SnakeEnv({"board_size": (args.height, args.width)}, render_mode="human")

    # Ensure window initializes
    obs, info = env.reset()
    env.render()

    try:
        import pygame
    except ImportError:
        pygame = None

    for ep in range(args.episodes):
        if ep > 0:
            obs, info = env.reset()
            env.render()
        done = False
        total_reward = 0.0
        while not done:
            # Pump events to keep window responsive
            if pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        env.close()
                        return

            action = algo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += float(reward)
            env.render()
            time.sleep(0.05)
        # short pause between episodes
        time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    main()
