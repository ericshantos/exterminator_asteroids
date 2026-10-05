import numpy as np
import pygame


class Bullet:
    RADIUS: int = 2
    SPEED: float = 10.0
    MAX_LIFETIME: int = 60

    def __init__(
        self,
        x: float,
        y: float,
        angle: float,
        inherit_vx: float = 0.0,
        inherit_vy: float = 0.0,
    ) -> None:
        self.x: float = x
        self.y: float = y

        radians: float = float(np.radians(angle))
        self.velocity_x: float = float(np.sin(radians)) * self.SPEED + inherit_vx
        self.velocity_y: float = -float(np.cos(radians)) * self.SPEED + inherit_vy

        self.lifetime: int = 0

    def update(self, width: int, height: int) -> None:
        self.x = (self.x + self.velocity_x) % width
        self.y = (self.y + self.velocity_y) % height

        self.lifetime += 1

    def is_alive(self) -> bool:
        return self.lifetime < self.MAX_LIFETIME

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(
            screen, (255, 255, 255), (int(self.x), int(self.y)), self.RADIUS
        )
