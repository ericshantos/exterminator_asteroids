from agent import PPOAgent
from configs import cfg

from .trainer import Trainer

if __name__ == "__main__":
    agent: PPOAgent = PPOAgent(config=cfg.ppo, seed=cfg.training.seed)

    trainer: Trainer = Trainer(agent, model_path="models/exterminator_ppo.zip")

    try:
        trainer.train(timesteps=cfg.ppo.total_timesteps)
    finally:
        agent.close()
