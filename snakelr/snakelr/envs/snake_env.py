from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class SnakeConfig:
    board_size: Tuple[int, int] = (15, 15)  # (height, width)
    step_penalty: float = -0.01
    food_reward: float = 1.0
    death_penalty: float = -1.0
    enable_poison: bool = False
    enable_obstacles: bool = False
    enable_multi_food: bool = False
    max_steps: Optional[int] = None
    # Rendering colors (RGB) in 0..255 for human mode; we scale for obs
    color_background: Tuple[int, int, int] = (0, 0, 0)
    color_snake_head: Tuple[int, int, int] = (0, 255, 0)
    color_snake_body: Tuple[int, int, int] = (0, 128, 0)
    color_food: Tuple[int, int, int] = (255, 0, 0)
    color_poison: Tuple[int, int, int] = (128, 0, 128)
    color_obstacle: Tuple[int, int, int] = (128, 128, 128)


Direction = Tuple[int, int]
UP: Direction = (-1, 0)
DOWN: Direction = (1, 0)
LEFT: Direction = (0, -1)
RIGHT: Direction = (0, 1)

ACTION_TO_DIR: Dict[int, Direction] = {
    0: UP,
    1: DOWN,
    2: LEFT,
    3: RIGHT,
}


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, config: Optional[SnakeConfig | Dict] = None, render_mode: Optional[str] = None):
        super().__init__()
        if config is None:
            config = SnakeConfig()
        if isinstance(config, dict):
            config = SnakeConfig(**config)
        self.cfg: SnakeConfig = config
        self.render_mode = render_mode

        h, w = self.cfg.board_size
        # Observation: HxWx3 float32 in [0,1]
        self.observation_space = spaces.Box(low=np.float32(0.0), high=np.float32(1.0), shape=(h, w, 3), dtype=np.float32)
        # Action: 0 up, 1 down, 2 left, 3 right
        self.action_space = spaces.Discrete(4)

        # Internal state
        self.rng = random.Random()
        self.board_height = h
        self.board_width = w
        self.snake: Deque[Tuple[int, int]] = deque()
        self.direction: Direction = RIGHT
        self.food_positions: List[Tuple[int, int]] = []
        self.poison_positions: List[Tuple[int, int]] = []
        self.obstacles: List[Tuple[int, int]] = []
        self.score: int = 0
        self.steps: int = 0
        self.done: bool = False

        # Rendering
        self._pygame_initialized: bool = False
        self._screen = None
        self._clock = None
        self._cell_px: int = 24

    def seed(self, seed: Optional[int] = None):
        self.rng.seed(seed)
        np.random.seed(seed)

    def _center_cell(self) -> Tuple[int, int]:
        return (self.board_height // 2, self.board_width // 2)

    def _spawn_food(self, count: int = 1):
        for _ in range(count):
            empty_cells = self._get_empty_cells()
            if not empty_cells:
                return
            pos = self.rng.choice(empty_cells)
            if pos not in self.food_positions:
                self.food_positions.append(pos)

    def _spawn_poison(self, count: int = 1):
        if not self.cfg.enable_poison:
            return
        for _ in range(count):
            empty_cells = self._get_empty_cells()
            if not empty_cells:
                return
            pos = self.rng.choice(empty_cells)
            if pos not in self.poison_positions:
                self.poison_positions.append(pos)

    def _spawn_obstacles(self, count: int = 0):
        if not self.cfg.enable_obstacles or count <= 0:
            return
        for _ in range(count):
            empty_cells = self._get_empty_cells()
            if not empty_cells:
                return
            pos = self.rng.choice(empty_cells)
            if pos not in self.obstacles:
                self.obstacles.append(pos)

    def _get_empty_cells(self) -> List[Tuple[int, int]]:
        occupied = set(self.snake) | set(self.food_positions) | set(self.poison_positions) | set(self.obstacles)
        return [
            (r, c)
            for r in range(self.board_height)
            for c in range(self.board_width)
            if (r, c) not in occupied
        ]

    def _reset_state(self):
        self.snake.clear()
        self.score = 0
        self.steps = 0
        self.done = False
        self.direction = RIGHT
        self.food_positions = []
        self.poison_positions = []
        self.obstacles = []

        start = self._center_cell()
        self.snake.append(start)

        if self.cfg.enable_obstacles:
            area = self.board_height * self.board_width
            self._spawn_obstacles(max(0, area // 20))

        if self.cfg.enable_multi_food:
            self._spawn_food(count=3)
        else:
            self._spawn_food(count=1)

        if self.cfg.enable_poison:
            self._spawn_poison(count=1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._reset_state()
        obs = self._get_observation()
        info: Dict = {"score": self.score}
        return obs, info

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Step called after episode is done. Call reset().")
        assert self.action_space.contains(action), f"Invalid action: {action}"

        proposed_dir = ACTION_TO_DIR[int(action)]
        if len(self.snake) > 1:
            head_r, head_c = self.snake[0]
            neck_r, neck_c = self.snake[1]
            current_dir = (head_r - neck_r, head_c - neck_c)
            if (proposed_dir[0] == -current_dir[0] and proposed_dir[1] == -current_dir[1]):
                proposed_dir = current_dir

        self.direction = proposed_dir

        reward = self.cfg.step_penalty
        self.steps += 1

        head_r, head_c = self.snake[0]
        dr, dc = self.direction
        new_head = (head_r + dr, head_c + dc)

        if (
            new_head[0] < 0 or new_head[0] >= self.board_height or
            new_head[1] < 0 or new_head[1] >= self.board_width
        ):
            self.done = True
            reward += self.cfg.death_penalty
            obs = self._get_observation()
            info = {"score": self.score, "reason": "wall"}
            return obs, reward, self.done, False, info

        if new_head in self.snake:
            self.done = True
            reward += self.cfg.death_penalty
            obs = self._get_observation()
            info = {"score": self.score, "reason": "self"}
            return obs, reward, self.done, False, info

        if new_head in self.obstacles:
            self.done = True
            reward += self.cfg.death_penalty
            obs = self._get_observation()
            info = {"score": self.score, "reason": "obstacle"}
            return obs, reward, self.done, False, info

        grew = False
        if new_head in self.food_positions:
            self.food_positions.remove(new_head)
            grew = True
            self.score += 1
            reward += self.cfg.food_reward
            self._spawn_food(count=1)

        if self.cfg.enable_poison and new_head in self.poison_positions:
            self.poison_positions.remove(new_head)
            reward += self.cfg.death_penalty
            self.done = True
            obs = self._get_observation()
            info = {"score": self.score, "reason": "poison"}
            return obs, reward, self.done, False, info

        self.snake.appendleft(new_head)
        if not grew:
            self.snake.pop()

        if self.cfg.max_steps is not None and self.steps >= self.cfg.max_steps:
            self.done = True

        obs = self._get_observation()
        info = {"score": self.score}
        return obs, reward, self.done, False, info

    def _get_observation(self) -> np.ndarray:
        h, w = self.board_height, self.board_width
        img = np.zeros((h, w, 3), dtype=np.float32)

        inv255 = np.float32(1.0 / 255.0)
        def norm(rgb):
            return np.array(rgb, dtype=np.float32) * inv255

        bg = norm(self.cfg.color_background)
        obs_color = norm(self.cfg.color_obstacle)
        food_color = norm(self.cfg.color_food)
        poison_color = norm(self.cfg.color_poison)
        head_color = norm(self.cfg.color_snake_head)
        body_color = norm(self.cfg.color_snake_body)

        img[:, :] = bg

        for r, c in self.obstacles:
            img[r, c] = obs_color
        for r, c in self.food_positions:
            img[r, c] = food_color
        for r, c in self.poison_positions:
            img[r, c] = poison_color
        for i, (r, c) in enumerate(self.snake):
            img[r, c] = head_color if i == 0 else body_color
        return img

    def render(self):
        mode = self.render_mode
        if mode == "rgb_array":
            return self._rgb_array()
        elif mode == "human":
            return self._render_pygame()
        else:
            return self._rgb_array()

    def _rgb_array(self) -> np.ndarray:
        obs = self._get_observation()
        return (obs * 255).clip(0, 255).astype(np.uint8)

    def _ensure_pygame(self):
        if self._pygame_initialized:
            return
        import pygame
        pygame.init()
        self._clock = pygame.time.Clock()
        height_px = self.board_height * self._cell_px
        width_px = self.board_width * self._cell_px
        self._screen = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("SnakeEnv")
        self._pygame_initialized = True

    def _render_pygame(self):
        import pygame
        self._ensure_pygame()
        surface = self._screen
        surface.fill(self.cfg.color_background)

        for r, c in self.obstacles:
            pygame.draw.rect(
                surface,
                self.cfg.color_obstacle,
                pygame.Rect(c * self._cell_px, r * self._cell_px, self._cell_px, self._cell_px),
            )
        for r, c in self.food_positions:
            pygame.draw.rect(
                surface,
                self.cfg.color_food,
                pygame.Rect(c * self._cell_px, r * self._cell_px, self._cell_px, self._cell_px),
            )
        for r, c in self.poison_positions:
            pygame.draw.rect(
                surface,
                self.cfg.color_poison,
                pygame.Rect(c * self._cell_px, r * self._cell_px, self._cell_px, self._cell_px),
            )

        for i, (r, c) in enumerate(self.snake):
            color = self.cfg.color_snake_head if i == 0 else self.cfg.color_snake_body
            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(c * self._cell_px, r * self._cell_px, self._cell_px, self._cell_px),
            )

        pygame.display.flip()
        if self.metadata.get("render_fps"):
            self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._pygame_initialized:
            import pygame
            pygame.quit()
            self._pygame_initialized = False
