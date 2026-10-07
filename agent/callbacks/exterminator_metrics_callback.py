from stable_baselines3.common.callbacks import BaseCallback


class ExterminatorMetricsCallback(BaseCallback):
    KEYS: dict[str, str] = {
        "score": "exterminator/score",
        "asteroid_destroyed": "exterminator/destroyed",
        "wave": "exterminator/wave",
        "accuracy": "exterminator/accuracy",
        "frame_count": "exterminator/frames",
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
