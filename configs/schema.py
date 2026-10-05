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
    max_steps_per_episode: int


@dataclass
class RewardConfig:
    player_died: float
    survive_step: float
    idle_penalty: float


@dataclass
class EnvironmentConfig:
    screen: ScreenConfig
    game: GameConfig
    reward: RewardConfig


@dataclass
class RLConfig:
    observation_radius: float
    max_episode_steps: int
    normalize_observation: bool


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
    target_update_interval: int
    total_timesteps: int
    exploration: ExplorationConfig


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
    training: TrainingConfig
