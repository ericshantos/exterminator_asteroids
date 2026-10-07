from dataclasses import dataclass


@dataclass
class ScreenConfig:
    width: int
    height: int
    fps: int
    window_title: str


@dataclass
class GameConfig:
    starting_lives: int
    extra_life_score: int
    initial_asteroids: int
    max_asteroids: int


@dataclass
class RewardConfig:
    survive_step: float
    score_scale: float
    life_lost: float
    game_over: float
    hyperspace_cost: float
    missed_shot: float
    danger_shaping: float
    aim_shaping: float


@dataclass
class EnvironmentConfig:
    screen: ScreenConfig
    game: GameConfig
    reward: RewardConfig


@dataclass
class RLConfig:
    max_episode_steps: int
    frame_skip: int


@dataclass
class ExplorationConfig:
    initial_eps: float
    final_eps: float
    fraction: float


@dataclass
class DQNConfig:
    learning_rate: float
    gamma: float
    buffer_size: int
    batch_size: int
    learning_starts: int
    train_freq: int
    gradient_steps: int
    target_update_interval: int
    total_timesteps: int
    exploration: ExplorationConfig


@dataclass
class PPOConfig:
    n_envs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    n_steps: int
    batch_size: int
    epochs: int
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    total_timesteps: int


@dataclass
class TrainingConfig:
    total_timesteps: int
    evaluation_frequency: int
    save_frequency: int
    seed: int
    device: str


@dataclass
class ApolloConfig:
    screen: ScreenConfig
    game: GameConfig
    reward: RewardConfig
    rl: RLConfig
    dqn: DQNConfig
    ppo: PPOConfig
    training: TrainingConfig
