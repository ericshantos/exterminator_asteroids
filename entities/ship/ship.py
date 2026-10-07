from abc import ABC

from ..bullet import Bullet, BulletManager


class Shooter(ABC):
    MAX_BULLETS: int = 4

    x: float
    y: float

    def __init__(self) -> None:
        self.bullet_manager = BulletManager()

        self.shots_fired: int = 0

    def can_shoot(self) -> bool:
        return self.bullet_manager.count < self.MAX_BULLETS

    def fire(
        self,
        x: float,
        y: float,
        angle: float,
        inherit_vx: float = 0.0,
        inherit_vy: float = 0.0,
    ) -> bool:
        if not self.can_shoot():
            return False

        self.shots_fired += 1

        self.bullet_manager.shoot(x, y, angle, inherit_vx, inherit_vy)

        return True

    @property
    def bullets(self) -> list[Bullet]:
        return self.bullet_manager.bullets
