from typing import Any, Protocol, cast

import gymnasium as gym
import numpy as np

from .action_space import Action

_RELEASED: dict[Action, Action] = {
    Action.SHOOT: Action.NOTHING,
    Action.HYPERSPACE: Action.NOTHING,
    Action.THRUST_SHOOT: Action.THRUST,
    Action.LEFT_SHOOT: Action.LEFT,
    Action.RIGHT_SHOOT: Action.RIGHT,
    Action.LEFT_THRUST_SHOOT: Action.LEFT_THRUST,
    Action.RIGHT_THRUST_SHOOT: Action.RIGHT_THRUST,
}


class Observable(Protocol):
    skip_observation: bool

    def observe(self) -> np.ndarray: ...


class FrameSkip(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env: gym.Env[np.ndarray, int], skip: int = 4) -> None:
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")

        super().__init__(env)

        self.skip = skip

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        first = Action(int(action))
        held = _RELEASED.get(first, first)

        base = cast(Observable, self.env.unwrapped)

        total_reward = 0.0

        base.skip_observation = True

        try:
            for frame in range(self.skip):
                _, reward, terminated, truncated, info = self.env.step(
                    first if frame == 0 else held
                )

                total_reward += float(reward)

                if terminated or truncated:
                    break
        finally:
            base.skip_observation = False

        return base.observe(), total_reward, terminated, truncated, info
