from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from ..callbacks import ApolloMetricsCallback

from configs.schema import PPOConfig
from env.observation import Observation
from env import make_env


class PPOAgent:
    def __init__(self, config: PPOConfig, seed: int | None = None) -> None:

        vec_env = SubprocVecEnv([make_env for _ in range(config.n_envs)])
        vec_env.seed(seed)

        self._env: VecNormalize = VecNormalize(
            vec_env, norm_obs=False, norm_reward=True, gamma=config.gamma
        )

        policy_kwargs: dict[str, dict[str, list[int]]] = dict(
            net_arch=dict(pi=[512, 512], vf=[512, 512])
        )

        initial_lr = config.learning_rate

        self._model: PPO = PPO(
            policy="MlpPolicy",
            env=self._env,
            learning_rate=lambda progress_remaining: initial_lr * progress_remaining,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            tensorboard_log="./logs/tensorboard",
            policy_kwargs=policy_kwargs,
            device="cpu",
            seed=seed,
            verbose=1,
        )

    def train(
        self,
        total_timesteps: int,
    ) -> None:
        callback = ApolloMetricsCallback()

        self._model.learn(
            total_timesteps=total_timesteps, callback=callback, tb_log_name="apollo_ppo"
        )

    def act(self, observation: Observation, deterministic: bool = True) -> int:
        action, _ = self._model.predict(observation, deterministic=deterministic)

        return int(action)

    def save(self, path: str | Path) -> None:
        self._model.save(path)

        self._env.save(str(Path(path).with_suffix(".vecnormalize.pkl")))

    def close(self) -> None:
        self._env.close()
