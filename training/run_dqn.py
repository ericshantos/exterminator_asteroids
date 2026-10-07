from agent import DQNAgent
from configs import cfg
from env import make_env

from .trainer import Trainer

env = make_env()

agent: DQNAgent = DQNAgent(env=env, config=cfg.dqn, seed=cfg.training.seed)

trainer: Trainer = Trainer(agent, model_path="models/apollo_dqn.zip")

trainer.train(timesteps=cfg.dqn.total_timesteps)
