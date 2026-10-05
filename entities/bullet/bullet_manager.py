import pygame

from .bullet import Bullet


class BulletManager:
    def __init__(self) -> None:
        self.bullets: list[Bullet] = []

    def shoot(
        self,
        x: float,
        y: float,
        angle: float,
        inherit_vx: float = 0.0,
        inherit_vy: float = 0.0,
    ) -> None:
        self.bullets.append(Bullet(x, y, angle, inherit_vx, inherit_vy))

    def update(self, width: int, height: int) -> None:
        for bullet in self.bullets:
            bullet.update(width, height)

        self.clean_expired_bullets()

    def remove(self, bullet: Bullet) -> None:
        if bullet in self.bullets:
            self.bullets.remove(bullet)

    def clear(self) -> None:
        self.bullets.clear()

    def clean_expired_bullets(self) -> None:
        self.bullets = [bullet for bullet in self.bullets if bullet.is_alive()]

    def draw(self, screen: pygame.Surface) -> None:
        for bullet in self.bullets:
            bullet.draw(screen)

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    @property
    def count(self) -> int:
        return len(self.bullets)
