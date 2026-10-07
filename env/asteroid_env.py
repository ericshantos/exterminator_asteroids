import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs import cfg

from .action_space import ActionSpace
from .game_world import GameWorld
from .observation import Observation
from .reward import RewardFunction
from .toroidal_space import ToroidalSpace
from rendering import Renderer


class AsteroidEnv(gym.Env[np.ndarray, int]):
    metadata: dict[str, list[str] | int] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.render_mode = render_mode

        self.space: ToroidalSpace = ToroidalSpace(
            cfg.screen.width, cfg.screen.height
        )

        self.world: GameWorld = GameWorld(self.space)

        self.observation: Observation = Observation(self.space)

        if render_mode == "human":
            self.renderer: Renderer = Renderer(self.world)

        self.reward_system: RewardFunction = RewardFunction(self.space)

        self.action_space: spaces.Discrete = spaces.Discrete(ActionSpace.N_ACTIONS)

        self.observation_space: spaces.Box = spaces.Box(
            low=-1, high=1, shape=(Observation.SIZE,), dtype=np.float32
        )

        self.current_step: int = 0
        self.max_episode_steps: int = cfg.rl.max_episode_steps

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # As entidades sorteiam com o `random` global, não com `self.np_random`.
        if seed is not None:
            random.seed(seed)

        self.current_step = 0
        self.world.reset()
        self.reward_system.reset()

        obs = self.observation.build(self.world)

        info: dict[str, int | float] = {}

        return obs, info

    def step(
        self, action_id: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.current_step += 1

        if self.render_mode == "human":
            running = self.renderer.handle_events()

            if not running:
                self.world.done = True

                obs = self.observation.build(self.world)

                return (obs, 0.0, True, False, {})

        action = ActionSpace.to_action(action_id)

        self.world.update(action)

        self.render()

        obs = self.observation.build(self.world)

        reward = self.reward_system.compute(self.world, action)

        terminated = self.world.is_done()

        truncated = False

        if self.current_step >= self.max_episode_steps:
            truncated = True

        info = {
            "score": self.world.score,
            "lives": self.world.player_lives,
            "wave": self.world.wave,
            "frame_count": self.world.frame_count,
            "asteroid_destroyed": self.world.asteroids_destroyed,
            "shots_fired": self.world.shots_fired,
            "accuracy_hits": self.world.accuracy_hits,
            "accuracy": self.world.accuracy,
            **self.world.diagnostics(),
        }

        return (obs, reward, terminated, truncated, info)

    def render(self) -> None:
        if self.render_mode != "human":
            return
        
        self.renderer.draw()

    def close(self) -> None:
        pass
