import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from configs import cfg

from .asteroid_env import AsteroidEnv
from .frame_skip import FrameSkip


def make_env(render_mode: str | None = None) -> gym.Env[np.ndarray, int]:
    return FrameSkip(Monitor(AsteroidEnv(render_mode=render_mode)), cfg.rl.frame_skip)
