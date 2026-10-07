from stable_baselines3.common.callbacks import BaseCallback


class ApolloMetricsCallback(BaseCallback):
    KEYS: dict[str, str] = {
        "score": "apollo/score",
        "asteroid_destroyed": "apollo/destroyed",
        "wave": "apollo/wave",
        "accuracy": "apollo/accuracy",
        "frame_count": "apollo/frames",
    }

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if not done:
                continue

            for key, name in self.KEYS.items():
                self.logger.record_mean(name, info.get(key, 0))

        return True
