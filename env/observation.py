import math

import numpy as np

from entities import Asteroid, AsteroidManager, Bullet, Player, Saucer

from .game_world import GameWorld
from .kinematics import (
    NO_COLLISION,
    RelativeState,
    angle_difference,
    closest_approach,
    escape_time_to_collision,
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

    RADAR_SECTORS: int = 16
    RADAR_FEATURES: int = RADAR_SECTORS + 3

    SIZE: int = (
        SHIP_FEATURES
        + SAUCER_FEATURES
        + MAX_SAUCER_BULLETS * BULLET_FEATURES
        + MAX_ASTEROIDS * TARGET_FEATURES
        + RADAR_FEATURES
    )

    # Horizonte de 2 s para normalizar tempos até a colisão.
    TTC_HORIZON: float = 120.0

    MAX_LIVES: int = 5

    TARGET_REL_SPEED: float = Player.MAX_SPEED + Asteroid.MAX_SPEED + Saucer.MAX_SPEED
    BULLET_REL_SPEED: float = Player.MAX_SPEED + Bullet.SPEED

    SAUCER_MAX_RADIUS: float = 24.0

    ESCAPE_FRAMES: int = 20
    ESCAPE_SPEED: float = Player.ACCELERATION * ESCAPE_FRAMES

    # Horário a partir do nariz: 0°, 22,5° ... 180°, -157,5° ... -22,5°.
    RADAR_ANGLES: tuple[float, ...] = tuple(
        math.remainder(float(angle), 2 * math.pi)
        for angle in np.arange(RADAR_SECTORS) * (2 * math.pi / RADAR_SECTORS)
    )

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

    def _encode_radar(
        self, own: RelativeState, threats: list[tuple[RelativeState, float]]
    ) -> list[float]:
        course = min(
            (time_to_collision(state, contact) for state, contact in threats),
            default=NO_COLLISION,
        )

        sectors: list[float] = []

        for angle in self.RADAR_ANGLES:
            turn_frames = abs(math.degrees(angle)) / Player.ROTATION_SPEED

            forward = own.vel_forward + self.ESCAPE_SPEED * math.cos(angle)
            right = own.vel_right + self.ESCAPE_SPEED * math.sin(angle)

            speed = math.hypot(forward, right)

            if speed > Player.MAX_SPEED:
                forward *= Player.MAX_SPEED / speed
                right *= Player.MAX_SPEED / speed

            gain_forward = forward - own.vel_forward
            gain_right = right - own.vel_right

            ttc = min(
                (
                    escape_time_to_collision(
                        state, contact, turn_frames, gain_forward, gain_right
                    )
                    for state, contact in threats
                ),
                default=NO_COLLISION,
            )

            sectors.append(self._norm_ttc(ttc))

        best_value = max(sectors)

        # Empate: a fatia mais perto do nariz; entre ±θ, a da direita.
        best = min(
            (k for k, value in enumerate(sectors) if value == best_value),
            key=lambda k: (abs(self.RADAR_ANGLES[k]), -self.RADAR_ANGLES[k]),
        )

        return [
            *sectors,
            self._norm_ttc(course),
            self.RADAR_ANGLES[best] / math.pi,
            best_value,
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

        threats: list[tuple[RelativeState, float]] = []

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

            threats.append((state, saucer.radius + Player.RADIUS))

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

        threats.extend(
            (state, Bullet.RADIUS + Player.RADIUS) for state in bullet_states
        )

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

        threats.extend(
            (state, radius + Player.RADIUS) for state, radius in asteroid_states
        )

        # Distância entre as bordas: um asteroide grande fica à frente de um
        # pequeno à mesma distância do centro.
        asteroid_states.sort(key=lambda item: item[0].distance - item[1])

        visible = asteroid_states[: self.MAX_ASTEROIDS]

        for state, radius in visible:
            obs.extend(self._encode_target(state, radius, Asteroid.MAX_RADIUS))

        obs.extend([0.0] * self.TARGET_FEATURES * (self.MAX_ASTEROIDS - len(visible)))

        obs.extend(self._encode_radar(own, threats))

        return np.asarray(obs, dtype=np.float32)
