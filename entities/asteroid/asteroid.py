from __future__ import annotations

import random
from typing import Literal, cast

import numpy as np
import pygame

from env.toroidal_space import ToroidalSpace

from .typed import Particle

AsteroidSize = Literal[1, 2, 3]

_SHAPES: tuple[tuple[tuple[float, float], ...], ...] = (
    ((-4, -2), (-2, -4), (0, -2), (2, -4), (4, -2), (3, 0),
     (4, 2), (1, 4), (-2, 4), (-4, 2)),
    ((-4, -2), (-2, -4), (2, -4), (4, -2), (4, -1), (1, 0),
     (4, 2), (2, 4), (1, 3), (-2, 4), (-4, 1), (-3, 0)),
    ((-4, -2), (-1, -4), (2, -4), (4, -1), (4, 1), (2, 4),
     (0, 4), (0, 1), (-2, 4), (-4, 1), (-2, -1)),
    ((-4, -1), (-2, -4), (0, -3), (2, -4), (4, -2), (2, -1),
     (4, 1), (2, 4), (-1, 3), (-2, 4), (-4, 2)),
)


class Asteroid:
    _RADIUS: dict[int, float] = {3: 40.0, 2: 20.0, 1: 10.0}

    _SPEED_RANGE: dict[int, tuple[float, float]] = {
        3: (0.75, 1.5),
        2: (1.2, 2.8),
        1: (1.8, 4.2),
    }

    MAX_RADIUS: float = _RADIUS[3]
    MAX_SPEED: float = _SPEED_RANGE[1][1]

    _ASTEROID_POINT: dict[int, int] = {1: 100, 2: 50, 3: 20}

    def __init__(
        self,
        space: ToroidalSpace,
        size: AsteroidSize = 3,
        x: float | None = None,
        y: float | None = None,
        velocity_x: float | None = None,
        velocity_y: float | None = None,
    ) -> None:
        self.space = space
        self.size: AsteroidSize = size
        self.radius: float = self._RADIUS[size]

        self.is_alive: bool = True
        self.particles: list[Particle] = []

        if x is None or y is None:
            x, y = self._generate_edge_position()

        self.x: float = space.wrap_x(x)
        self.y: float = space.wrap_y(y)

        if velocity_x is None or velocity_y is None:
            velocity_x, velocity_y = self._random_velocity()

        self.velocity_x: float = velocity_x
        self.velocity_y: float = velocity_y

        scale = self.radius / 4
        template = random.choice(_SHAPES)
        self.shape: list[tuple[float, float]] = [
            (px * scale, py * scale) for px, py in template
        ]

    def _generate_edge_position(self) -> tuple[float, float]:
        if random.random() < 0.5:
            return 0.0, random.uniform(0, self.space.height)

        return random.uniform(0, self.space.width), 0.0

    def _random_velocity(self) -> tuple[float, float]:
        low, high = self._SPEED_RANGE[self.size]

        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(low, high)

        return float(np.cos(angle)) * speed, float(np.sin(angle)) * speed

    def _clamp_speed(self, vx: float, vy: float) -> tuple[float, float]:
        low, high = self._SPEED_RANGE[self.size]

        speed = float(np.hypot(vx, vy))

        if speed < 1e-6:
            return self._random_velocity()

        clamped = min(max(speed, low), high)

        return vx / speed * clamped, vy / speed * clamped

    def trigger_explosion(self) -> None:
        self.is_alive = False
        for _ in range(random.randint(8, 15)):
            angle = random.uniform(0, np.pi * 2)
            speed = random.uniform(1.0, 3.5)
            self.particles.append(
                {
                    "x": self.x,
                    "y": self.y,
                    "vel_x": np.cos(angle) * speed,
                    "vel_y": np.sin(angle) * speed,
                    "alpha": 255,
                }
            )

    @property
    def explosion_finished(self) -> bool:
        return all(p["alpha"] <= 0 for p in self.particles)

    def update(self) -> None:
        if not self.is_alive:
            for p in self.particles:
                p["x"] += p["vel_x"]
                p["y"] += p["vel_y"]
                p["alpha"] = max(0, p["alpha"] - 6)
            return

        self.x = self.space.wrap_x(self.x + self.velocity_x)
        self.y = self.space.wrap_y(self.y + self.velocity_y)

    def create_children(self, quantity: int) -> list[Asteroid]:
        if self.size == 1:
            return []

        child_size = cast(AsteroidSize, self.size - 1)

        children = []

        for _ in range(quantity):
            child = Asteroid(space=self.space, size=child_size, x=self.x, y=self.y)

            kick_x, kick_y = child.velocity_x, child.velocity_y

            child.velocity_x, child.velocity_y = child._clamp_speed(
                self.velocity_x + kick_x,
                self.velocity_y + kick_y,
            )

            children.append(child)

        return children

    def draw(self, screen: pygame.Surface) -> None:
        if not self.is_alive:
            for p in self.particles:
                if p["alpha"] > 0:
                    color = (p["alpha"], p["alpha"], p["alpha"])
                    pygame.draw.circle(screen, color, (int(p["x"]), int(p["y"])), 1)
            return

        for offset_x, offset_y in self.space.ghost_offsets(
            self.x, self.y, self.radius
        ):
            transformed_points = [
                (self.x + offset_x + px, self.y + offset_y + py)
                for px, py in self.shape
            ]
            pygame.draw.polygon(screen, (255, 255, 255), transformed_points, 2)

    @classmethod
    def points(cls, size: int) -> int:
        return cls._ASTEROID_POINT.get(size, 0)
