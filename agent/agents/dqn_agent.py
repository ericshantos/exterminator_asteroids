from pathlib import Path

from stable_baselines3 import DQN

from ..callbacks import ExterminatorMetricsCallback

import gymnasium as gym
import numpy as np

from configs.schema import DQNConfig
from env.observation import Observation
from training import DeviceManager



class DQNAgent:
    def __init__(
        self,
        env: gym.Env[np.ndarray, int],
        config: DQNConfig,
        seed: int | None = None,
    ) -> None:

        device: DeviceManager = DeviceManager(prefer_gpu=True).get_device()

        policy_kwargs: dict[str, list[int]] = dict(net_arch=[256, 256, 256])

        self._model: DQN = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            gamma=config.gamma,
            tensorboard_log="./logs/tensorboard",
            learning_starts=config.learning_starts,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            exploration_initial_eps=config.exploration.initial_eps,
            exploration_final_eps=config.exploration.final_eps,
            exploration_fraction=config.exploration.fraction,
            policy_kwargs=policy_kwargs,
            device=device,
            target_update_interval=config.target_update_interval,
            seed=seed,
            verbose=1,
        )

    def train(
        self,
        total_timesteps: int,
    ) -> None:
        callback = ExterminatorMetricsCallback()

        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="exterminator_dqn",
        )

    def act(self, observation: Observation, deterministic: bool = True) -> int:
        action, _ = self._model.predict(observation, deterministic=deterministic)

        return int(action)

    def save(self, path: str | Path) -> None:
        self._model.save(path)
