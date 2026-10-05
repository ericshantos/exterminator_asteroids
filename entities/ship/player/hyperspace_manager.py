import random


class HyperspaceManager:
    DURATION: int = 48

    BASE_FAILURE_CHANCE: float = 0.04
    FAILURE_PER_ASTEROID: float = 0.006
    MAX_FAILURE_CHANCE: float = 0.25

    def __init__(self) -> None:
        self.timer: int = 0

    @property
    def is_active(self) -> bool:
        return self.timer > 0

    def enter(self) -> None:
        self.timer = self.DURATION

    def tick(self) -> bool:
        if self.timer <= 0:
            return False

        self.timer -= 1

        return self.timer == 0

    def teleport(
        self,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        return (random.uniform(0, width), random.uniform(0, height))

    def failure_chance(self, asteroid_count: int) -> float:
        chance = self.BASE_FAILURE_CHANCE + asteroid_count * self.FAILURE_PER_ASTEROID

        return min(chance, self.MAX_FAILURE_CHANCE)

    def should_explode(self, asteroid_count: int) -> bool:
        return random.random() < self.failure_chance(asteroid_count)

    def reset(self) -> None:
        self.timer = 0
