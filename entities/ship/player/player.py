import random

import numpy as np
import pygame

from configs import cfg
from env.action_space import ActionMap
from env.toroidal_space import ToroidalSpace

from ..destroyed import Destroyable
from ..ship import Shooter
from .hyperspace_manager import HyperspaceManager
from .ship_explosion_effect import ShipExplosionEffect


class Player(Shooter, Destroyable):
    MAX_SPEED: float = 10.0
    ROTATION_SPEED: float = 360.0 * 3 / 256
    ACCELERATION: float = 0.15
    FRICTION: float = 0.99

    RADIUS: int = 8
    NOSE_DISTANCE: float = 12.0

    STARTING_LIVES: int = cfg.game.starting_lives

    RESPAWN_DELAY: int = 90

    def __init__(self, x: float, y: float, space: ToroidalSpace) -> None:
        super().__init__()

        self.start_x: float = x
        self.start_y: float = y

        self.x = x
        self.y = y

        self.space = space

        self.angle: float = 0
        self.velocity_x: float = 0
        self.velocity_y: float = 0

        self.is_accelerating: bool = False

        self.used_hyperspace_this_step: bool = False
        self.fired_this_step: bool = False

        self._fire_held: bool = False
        self._hyperspace_held: bool = False

        self.is_alive: bool = True
        self._lives: int = self.STARTING_LIVES

        self.explosion: ShipExplosionEffect = ShipExplosionEffect()
        self.hyperspace_manager: HyperspaceManager = HyperspaceManager()

        self.respawn_timer: int = 0

    @property
    def lives(self) -> int:
        return self._lives

    @lives.setter
    def lives(self, value: int) -> None:
        self._lives = max(0, value)

    def gain_life(self) -> None:
        self._lives += 1

    def lose_life(self) -> None:
        self._lives = max(0, self._lives - 1)

    @property
    def in_hyperspace(self) -> bool:
        return self.hyperspace_manager.is_active

    @property
    def is_active(self) -> bool:
        return self.is_alive and not self.in_hyperspace

    @property
    def heading(self) -> tuple[float, float]:
        radians = float(np.radians(self.angle))

        return float(np.sin(radians)), -float(np.cos(radians))

    def update(self, action: ActionMap, asteroid_count: int = 0) -> None:
        self.used_hyperspace_this_step = False
        self.fired_this_step = False

        self.bullet_manager.update(self.space.width, self.space.height)

        fire_pressed = action.shoot and not self._fire_held
        hyperspace_pressed = action.hyperspace and not self._hyperspace_held

        self._fire_held = action.shoot
        self._hyperspace_held = action.hyperspace

        if not self.is_alive:
            self.explosion.update()

            if self.lives > 0:
                self.respawn_timer += 1

            return

        if self.in_hyperspace:
            if self.hyperspace_manager.tick():
                self._exit_hyperspace(asteroid_count)

            return

        if action.rotate_left:
            self.angle -= self.ROTATION_SPEED
        elif action.rotate_right:
            self.angle += self.ROTATION_SPEED

        self.angle %= 360.0

        self.is_accelerating = action.thrust

        if action.thrust:
            heading_x, heading_y = self.heading
            self.velocity_x += heading_x * self.ACCELERATION
            self.velocity_y += heading_y * self.ACCELERATION

            speed = float(np.hypot(self.velocity_x, self.velocity_y))

            if speed > self.MAX_SPEED:
                scale = self.MAX_SPEED / speed

                self.velocity_x *= scale
                self.velocity_y *= scale
        else:
            self.velocity_x *= self.FRICTION
            self.velocity_y *= self.FRICTION

        self.x = self.space.wrap_x(self.x + self.velocity_x)
        self.y = self.space.wrap_y(self.y + self.velocity_y)

        if fire_pressed:
            self.shoot()

        if hyperspace_pressed:
            self._enter_hyperspace()

    def shoot(self) -> None:
        if not self.is_active:
            return

        heading_x, heading_y = self.heading

        nose_x = self.space.wrap_x(self.x + heading_x * self.NOSE_DISTANCE)
        nose_y = self.space.wrap_y(self.y + heading_y * self.NOSE_DISTANCE)

        self.fired_this_step = self.fire(
            nose_x, nose_y, self.angle, self.velocity_x, self.velocity_y
        )

    def _enter_hyperspace(self) -> None:
        self.hyperspace_manager.enter()

        self.velocity_x = 0
        self.velocity_y = 0
        self.is_accelerating = False

        self.used_hyperspace_this_step = True

    def _exit_hyperspace(self, asteroid_count: int) -> None:
        self.x, self.y = self.hyperspace_manager.teleport(
            self.space.width,
            self.space.height,
        )

        if self.hyperspace_manager.should_explode(asteroid_count):
            self.die()

    def die(self) -> None:
        if not self.is_alive:
            return

        self.is_alive = False
        self.lose_life()

        self.is_accelerating = False
        self.hyperspace_manager.reset()

        self.explosion.trigger(
            self.x, self.y, self.angle, self.velocity_x, self.velocity_y
        )

        self.respawn_timer = 0

    def respawn(self) -> None:
        self.x = self.start_x
        self.y = self.start_y
        self.angle = 0
        self.velocity_x = 0
        self.velocity_y = 0

        self.hyperspace_manager.reset()

        self.explosion.reset()
        self.is_alive = True
        self.is_accelerating = False

    def draw(self, screen: pygame.Surface) -> None:
        self.bullet_manager.draw(screen)

        if not self.is_alive:
            self.explosion.draw(screen)
            return

        if self.in_hyperspace:
            return

        for offset_x, offset_y in self.space.ghost_offsets(
            self.x, self.y, self.NOSE_DISTANCE
        ):
            self._draw_ship(screen, self.x + offset_x, self.y + offset_y)

    def _draw_ship(self, screen: pygame.Surface, x: float, y: float) -> None:
        radians = float(np.radians(self.angle))

        tip_x = x + float(np.sin(radians)) * 12
        tip_y = y - float(np.cos(radians)) * 12
        right_x = x + float(np.sin(radians - 2.5)) * 10
        right_y = y - float(np.cos(radians - 2.5)) * 10
        inner_x = x - float(np.sin(radians)) * 4
        inner_y = y + float(np.cos(radians)) * 4
        left_x = x + float(np.sin(radians + 2.5)) * 10
        left_y = y - float(np.cos(radians + 2.5)) * 10

        pygame.draw.polygon(
            screen,
            (255, 255, 255),
            [(tip_x, tip_y), (right_x, right_y), (inner_x, inner_y), (left_x, left_y)],
            2,
        )

        if self.is_accelerating and random.choice([True, False]):
            fire_tip_x = x - float(np.sin(radians)) * 10
            fire_tip_y = y + float(np.cos(radians)) * 10
            fire_left_x = x + float(np.sin(radians + 2.8)) * 6
            fire_left_y = y - float(np.cos(radians + 2.8)) * 6
            fire_right_x = x + float(np.sin(radians - 2.8)) * 6
            fire_right_y = y - float(np.cos(radians - 2.8)) * 6

            pygame.draw.polygon(
                screen,
                (255, 255, 255),
                [
                    (inner_x, inner_y),
                    (fire_left_x, fire_left_y),
                    (fire_tip_x, fire_tip_y),
                    (fire_right_x, fire_right_y),
                ],
                2,
            )
