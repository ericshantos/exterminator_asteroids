from agent import AgentProtocol


class Trainer:
    def __init__(self, agent: AgentProtocol, model_path: str) -> None:
        self._agent = agent
        self._model_path = model_path

    def train(self, timesteps: int) -> None:
        self._agent.train(total_timesteps=timesteps)

        self._agent.save(self._model_path)
