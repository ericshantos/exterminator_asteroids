from stable_baselines3.common.callbacks import BaseCallback

from entities import DeathCause


class ExterminatorMetricsCallback(BaseCallback):
    KEYS: dict[str, str] = {
        "score": "exterminator/score",
        "asteroid_destroyed": "exterminator/destroyed",
        "wave": "exterminator/wave",
        "accuracy": "exterminator/accuracy",
        "frame_count": "exterminator/frames",
        "lives_lost": "exterminator/lives_lost",
        "points_per_life": "exterminator/points_per_life",
        "saucers_spawned": "exterminator/saucers_spawned",
        "saucers_destroyed": "exterminator/saucers_destroyed",
        "hyperspace_jumps": "exterminator/hyperspace_jumps",
        **{
            f"deaths_{cause.value}": f"exterminator/deaths/{cause.value}"
            for cause in DeathCause
        },
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
