import random
from typing import Literal

import numpy as np
import pygame

from env.toroidal_space import ToroidalSpace

from ..destroyed import Destroyable
from ..ship import Shooter
from .saucer_explosion_effect import SaucerExplosionEffect


class Saucer(Shooter, Destroyable):
    MAX_BULLETS: int = 2
    FIRE_INTERVAL: int = 30

    LARGE_SPEED: float = 2.0
    SMALL_SPEED: float = 3.5
    VERTICAL_SPEED: float = 2.0

    MAX_SPEED: float = float(np.hypot(SMALL_SPEED, VERTICAL_SPEED))

    _SAUCER_POINT: dict[str, int] = {
        "large": 200,
        "small": 1000,
    }

    def __init__(
        self,
        space: ToroidalSpace,
        size_type: Literal["large", "small"] = "large",
        aim_error: float = 45.0,
    ) -> None:
        super().__init__()

        self.space = space
        self.size_type = size_type

        self.aim_error = aim_error

        self.scale: float = 2.0 if size_type == "large" else 1.0

        self.color: tuple[int, int, int] = (255, 255, 255)

        self.raw_points: list[tuple[float, float]] = [
            (-9, 0),
            (-3, -3),
            (-2, -6),
            (2, -6),
            (3, -3),
            (9, 0),
            (3, 4),
            (-3, 4),
            (-9, 0),
            (9, 0),
            (3, -3),
            (-3, -3),
            (-9, 0),
        ]
        self.shape_points: list[tuple[float, float]] = [
            (x * self.scale, y * self.scale) for x, y in self.raw_points
        ]

        self.direction: Literal[-1, 1] = random.choice([-1, 1])
        self.x: float = 0.0 if self.direction == 1 else float(self.space.width)
        self.y: float = random.uniform(0, self.space.height)

        horizontal = self.LARGE_SPEED if size_type == "large" else self.SMALL_SPEED
        self.speed_x: float = horizontal * self.direction
        self.speed_y: float = self._random_vertical_speed()

        self.distance_travelled: float = 0.0

        self.dir_change_timer: int = self._random_dir_change_interval()
        self.fire_timer: int = self.FIRE_INTERVAL

        self.is_alive: bool = True

        self.explosion: SaucerExplosionEffect = SaucerExplosionEffect()

        self.is_exploding: bool = False

    @property
    def velocity_x(self) -> float:
        return self.speed_x

    @property
    def velocity_y(self) -> float:
        return self.speed_y

    def _random_vertical_speed(self) -> float:
        return random.choice([-self.VERTICAL_SPEED, 0.0, self.VERTICAL_SPEED])

    @staticmethod
    def _random_dir_change_interval() -> int:
        return random.randint(90, 180)

    def move(
        self, player_x: float | None = None, player_y: float | None = None
    ) -> None:
        if self.is_alive:
            self.x += self.speed_x
            self.distance_travelled += abs(self.speed_x)

            self.dir_change_timer -= 1

            if self.dir_change_timer <= 0:
                self.speed_y = self._random_vertical_speed()
                self.dir_change_timer = self._random_dir_change_interval()

            self.y = self.space.wrap_y(self.y + self.speed_y)

            if self.distance_travelled >= self.space.width:
                self.is_alive = False

            self.fire_timer -= 1

            if self.is_alive and self.fire_timer <= 0:
                self.shoot(player_x, player_y)
                self.fire_timer = self.FIRE_INTERVAL

        if self.is_exploding:
            self.explosion.update()

            if not self.explosion.is_active:
                self.is_exploding = False

        self.bullet_manager.update(self.space.width, self.space.height)

    def die(self) -> None:
        if not self.is_alive:
            return

        self.is_alive = False
        self.is_exploding = True

        self.explosion.trigger(self.x, self.y)

    def shoot(self, player_x: float | None, player_y: float | None) -> None:
        if not self.is_alive:
            return

        if player_x is None or player_y is None:
            return

        if self.size_type == "large":
            angle = random.uniform(0, 360)

        else:
            dx, dy = self.space.relative_vector(self.x, self.y, player_x, player_y)

            angle = float(np.degrees(np.arctan2(dx, -dy)))

            angle += random.gauss(0, self.aim_error)

        self.fire(self.x, self.y, angle)

    def draw(self, surface: pygame.Surface) -> None:
        self.bullet_manager.draw(surface)

        if self.is_exploding:
            self.explosion.draw(surface)

        if not self.is_alive:
            return

        for offset_x, offset_y in self.space.ghost_offsets(
            self.x, self.y, self.radius
        ):
            transformed_points = [
                (px + self.x + offset_x, py + self.y + offset_y)
                for px, py in self.shape_points
            ]
            pygame.draw.aalines(surface, self.color, True, transformed_points)

    @property
    def radius(self) -> float:
        return 12 * self.scale

    @classmethod
    def points(cls, size_type: str) -> int:
        return cls._SAUCER_POINT.get(size_type, 0)

    @property
    def can_be_removed(self) -> bool:
        return (
            not self.is_alive and not self.is_exploding and self.bullet_manager.is_empty
        )
