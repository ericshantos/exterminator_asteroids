import random
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from configs import cfg
from env.action_space import ActionSpace
from env.game_world import GameWorld
from env.observation import Observation
from env.toroidal_space import ToroidalSpace
from rendering import Renderer


class ArenaEnv(gym.Env[np.ndarray, int]):
    def __init__(self, render: bool = False) -> None:
        super().__init__()

        self.space = ToroidalSpace(cfg.screen.width, cfg.screen.height)
        self.world = GameWorld(self.space)
        self.observation = Observation(self.space)

        self.renderer: Renderer | None = Renderer(self.world) if render else None

        self.action_space: spaces.Discrete = spaces.Discrete(ActionSpace.N_ACTIONS)
        self.observation_space: spaces.Box = spaces.Box(
            low=-1, high=1, shape=(Observation.SIZE,), dtype=np.float32
        )

        self.closed_by_user = False
        self.peak_lives = 0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        self.world.reset()

        self.closed_by_user = False
        self.peak_lives = self.world.player_lives

        return self.observation.build(self.world), {}

    def step(
        self, action_id: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.renderer is not None and not self.renderer.handle_events():
            self.closed_by_user = True

            return self.observation.build(self.world), 0.0, True, False, {}

        self.world.update(ActionSpace.to_action(int(action_id)))

        self.peak_lives = max(self.peak_lives, self.world.player_lives)

        if self.renderer is not None:
            self.renderer.draw()

        return self.observation.build(self.world), 0.0, self.world.is_done(), False, {}

    @property
    def extra_lives(self) -> int:
        return (
            self.world.player_lives
            + self.world.lives_lost
            - self.world.player.STARTING_LIVES
        )

    def close(self) -> None:
        if self.renderer is not None:
            pygame.quit()
            self.renderer = None
