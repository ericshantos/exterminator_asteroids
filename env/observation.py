import math

import numpy as np

from entities import Asteroid, AsteroidManager, Bullet, Player, Saucer

from .game_world import GameWorld
from .kinematics import (
    NO_COLLISION,
    RelativeState,
    angle_difference,
    closest_approach,
    intercept,
    ship_axes,
    time_to_collision,
    to_ship_frame,
)
from .toroidal_space import ToroidalSpace


class Observation:
    MAX_ASTEROIDS: int = 8
    MAX_SAUCER_BULLETS: int = Saucer.MAX_BULLETS

    SHIP_FEATURES: int = 7
    TARGET_FEATURES: int = 12
    SAUCER_FEATURES: int = TARGET_FEATURES + 1
    BULLET_FEATURES: int = 6

    SIZE: int = (
        SHIP_FEATURES
        + SAUCER_FEATURES
        + MAX_SAUCER_BULLETS * BULLET_FEATURES
        + MAX_ASTEROIDS * TARGET_FEATURES
    )

    # Horizonte de 2 s para normalizar tempos até a colisão.
    TTC_HORIZON: float = 120.0

    MAX_LIVES: int = 5

    TARGET_REL_SPEED: float = Player.MAX_SPEED + Asteroid.MAX_SPEED + Saucer.MAX_SPEED
    BULLET_REL_SPEED: float = Player.MAX_SPEED + Bullet.SPEED

    SAUCER_MAX_RADIUS: float = 24.0

    def __init__(self, space: ToroidalSpace) -> None:
        self.space = space

        # Maior distância toroidal possível.
        self.max_distance = math.hypot(space.width / 2, space.height / 2)

    def _relative(
        self,
        player: Player,
        heading: tuple[float, float],
        right: tuple[float, float],
        x: float,
        y: float,
        vx: float,
        vy: float,
    ) -> RelativeState:
        dx, dy = self.space.relative_vector(player.x, player.y, x, y)

        return to_ship_frame(
            dx, dy, vx - player.velocity_x, vy - player.velocity_y, heading, right
        )

    def _norm_ttc(self, ttc: float) -> float:
        if ttc == NO_COLLISION:
            return 1.0

        return min(ttc / self.TTC_HORIZON, 1.0)

    def _encode_target(
        self, state: RelativeState, radius: float, max_radius: float
    ) -> list[float]:
        contact = radius + Player.RADIUS

        gap = max(0.0, state.distance - contact)
        miss = max(0.0, closest_approach(state) - contact)

        shot = intercept(state, Bullet.SPEED, Bullet.MAX_LIFETIME)

        hit_if_fire = 0.0

        if shot.reachable:
            travel = max(Bullet.SPEED * shot.flight_time, 1.0)
            tolerance = math.atan2(radius + Bullet.RADIUS, travel)

            if abs(angle_difference(shot.lead_angle, 0.0)) <= tolerance:
                hit_if_fire = 1.0

        speed = self.TARGET_REL_SPEED

        return [
            1.0,
            state.forward / self.max_distance,
            state.right / self.max_distance,
            min(gap / self.max_distance, 1.0),
            float(np.clip(state.vel_forward / speed, -1.0, 1.0)),
            float(np.clip(state.vel_right / speed, -1.0, 1.0)),
            min(radius / max_radius, 1.0),
            self._norm_ttc(time_to_collision(state, contact)),
            min(miss / self.max_distance, 1.0),
            shot.lead_angle / math.pi,
            1.0 if shot.reachable else 0.0,
            hit_if_fire,
        ]

    def _encode_bullet(self, state: RelativeState) -> list[float]:
        speed = self.BULLET_REL_SPEED
        contact = Player.RADIUS + Bullet.RADIUS

        return [
            1.0,
            state.forward / self.max_distance,
            state.right / self.max_distance,
            float(np.clip(state.vel_forward / speed, -1.0, 1.0)),
            float(np.clip(state.vel_right / speed, -1.0, 1.0)),
            self._norm_ttc(time_to_collision(state, contact)),
        ]

    def build(self, world: GameWorld) -> np.ndarray:
        player = world.player

        heading, right = ship_axes(player.angle)

        own = to_ship_frame(
            0.0, 0.0, player.velocity_x, player.velocity_y, heading, right
        )

        bullets_free = Player.MAX_BULLETS - player.bullet_manager.count

        obs: list[float] = [
            own.vel_forward / Player.MAX_SPEED,
            own.vel_right / Player.MAX_SPEED,
            1.0 if player.is_alive else 0.0,
            1.0 if player.in_hyperspace else 0.0,
            bullets_free / Player.MAX_BULLETS,
            min(player.lives, self.MAX_LIVES) / self.MAX_LIVES,
            len(world.asteroids) / AsteroidManager.MAX_ASTEROIDS,
        ]

        saucer = world.saucer

        if saucer is not None and saucer.is_alive:
            state = self._relative(
                player,
                heading,
                right,
                saucer.x,
                saucer.y,
                saucer.velocity_x,
                saucer.velocity_y,
            )

            obs.extend(
                self._encode_target(state, saucer.radius, self.SAUCER_MAX_RADIUS)
            )
            obs.append(1.0 if saucer.size_type == "small" else 0.0)
        else:
            obs.extend([0.0] * self.SAUCER_FEATURES)

        bullet_states: list[RelativeState] = []

        # Os tiros do disco continuam ativos mesmo depois que ele é destruído.
        if saucer is not None:
            bullet_states = [
                self._relative(
                    player, heading, right, b.x, b.y, b.velocity_x, b.velocity_y
                )
                for b in saucer.bullets
            ]

        bullet_states.sort(key=lambda s: s.distance)

        for state in bullet_states[: self.MAX_SAUCER_BULLETS]:
            obs.extend(self._encode_bullet(state))

        missing_bullets = self.MAX_SAUCER_BULLETS - min(
            len(bullet_states), self.MAX_SAUCER_BULLETS
        )
        obs.extend([0.0] * self.BULLET_FEATURES * missing_bullets)

        asteroid_states = [
            (
                self._relative(
                    player,
                    heading,
                    right,
                    a.x,
                    a.y,
                    a.velocity_x,
                    a.velocity_y,
                ),
                a.radius,
            )
            for a in world.asteroids
        ]

        # Distância entre as bordas: um asteroide grande fica à frente de um
        # pequeno à mesma distância do centro.
        asteroid_states.sort(key=lambda item: item[0].distance - item[1])

        visible = asteroid_states[: self.MAX_ASTEROIDS]

        for state, radius in visible:
            obs.extend(self._encode_target(state, radius, Asteroid.MAX_RADIUS))

        obs.extend([0.0] * self.TARGET_FEATURES * (self.MAX_ASTEROIDS - len(visible)))

        return np.asarray(obs, dtype=np.float32)
