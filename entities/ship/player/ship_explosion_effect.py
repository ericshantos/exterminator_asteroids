import random

import numpy as np
import pygame

from ..explosion_effect import ExplosionEffect
from .typed import Fragments


class ShipExplosionEffect(ExplosionEffect):
    def __init__(self) -> None:
        self.fragments: list[Fragments] = []
        self.is_active = False

    def trigger(
        self, x: float, y: float, angle: float, velocity_x: float, velocity_y: float
    ) -> None:
        self.fragments.clear()
        self.is_active = True

        ship_lines = [
            ((0, -12), (7, 10)),
            ((7, 10), (0, 5)),
            ((0, 5), (-7, 10)),
            ((-7, 10), (0, -12)),
        ]

        radians = float(np.radians(angle))
        cos_a = float(np.cos(radians))
        sin_a = float(np.sin(radians))

        for p1, p2 in ship_lines:
            x1 = x + (p1[0] * cos_a - p1[1] * sin_a)
            y1 = y + (p1[0] * sin_a + p1[1] * cos_a)

            x2 = x + (p2[0] * cos_a - p2[1] * sin_a)
            y2 = y + (p2[0] * sin_a + p2[1] * cos_a)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            dx = center_x - x
            dy = center_y - y

            dist = float(np.hypot(dx, dy)) + 0.1

            self.fragments.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "vel_x": (dx / dist) * random.uniform(0.3, 0.8) + velocity_x * 0.2,
                    "vel_y": (dy / dist) * random.uniform(0.3, 0.8) + velocity_y * 0.2,
                    "alpha": 255,
                }
            )

    def update(self) -> None:
        active_fragments = 0

        for frag in self.fragments:
            frag["x1"] += frag["vel_x"]
            frag["y1"] += frag["vel_y"]

            frag["x2"] += frag["vel_x"]
            frag["y2"] += frag["vel_y"]

            frag["alpha"] = max(0, frag["alpha"] - 4)

            if frag["alpha"] > 0:
                active_fragments += 1

        self.is_active = active_fragments > 0

    def draw(self, surface: pygame.Surface) -> None:
        for frag in self.fragments:
            if frag["alpha"] <= 0:
                continue

            color = (
                frag["alpha"],
                frag["alpha"],
                frag["alpha"],
            )

            pygame.draw.line(
                surface,
                color,
                (frag["x1"], frag["y1"]),
                (frag["x2"], frag["y2"]),
                2,
            )

    def reset(self) -> None:
        self.fragments.clear()
        self.is_active = False
