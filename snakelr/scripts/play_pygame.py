from __future__ import annotations

import time

import pygame

from snakelr.envs.snake_env import SnakeEnv


KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
}


def main():
    env = SnakeEnv({"board_size": (15, 15)}, render_mode="human")
    obs, info = env.reset()
    env.render()  # ensure pygame/video initialized before event loop
    done = False
    last_action = 3  # start moving right

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    return
                if event.key in KEY_TO_ACTION:
                    action = KEY_TO_ACTION[event.key]

        if action is None:
            action = last_action
        else:
            last_action = action

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        if done:
            time.sleep(0.4)
            obs, info = env.reset()
            env.render()
            last_action = 3

        pygame.time.wait(80)


if __name__ == "__main__":
    main()
