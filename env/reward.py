import math

from configs import cfg
from entities import Bullet, Player

from .action_space import ActionMap
from .game_world import GameWorld
from .kinematics import (
    NO_COLLISION,
    RelativeState,
    intercept,
    ship_axes,
    time_to_collision,
    to_ship_frame,
)
from .toroidal_space import ToroidalSpace


class RewardFunction:
    THREAT_HORIZON: float = 120.0

    def __init__(self, space: ToroidalSpace) -> None:
        self.space = space

        self.reset()

    def reset(self) -> None:
        self.previous_score = 0
        self.previous_lives = cfg.game.starting_lives
        self.previous_misses = 0
        self.previous_threat = 0.0
        self.previous_aim = 0.0
        self.previous_active: bool = True
        self.previous_target: int | None = None

    def _relative(
        self, world: GameWorld, x: float, y: float, vx: float, vy: float
    ) -> RelativeState:
        player = world.player
        heading, right = ship_axes(player.angle)

        dx, dy = self.space.relative_vector(player.x, player.y, x, y)

        return to_ship_frame(
            dx, dy, vx - player.velocity_x, vy - player.velocity_y, heading, right
        )

    def threat(self, world: GameWorld) -> float:
        if not world.player.is_active:
            return 0.0

        objects = [(a.x, a.y, a.velocity_x, a.velocity_y, a.radius) for a in world.asteroids]

        saucer = world.saucer

        if saucer is not None:
            if saucer.is_alive:
                objects.append(
                    (saucer.x, saucer.y, saucer.velocity_x, saucer.velocity_y, saucer.radius)
                )

            objects.extend(
                (b.x, b.y, b.velocity_x, b.velocity_y, Bullet.RADIUS) for b in saucer.bullets
            )

        nearest = NO_COLLISION

        for x, y, vx, vy, radius in objects:
            state = self._relative(world, x, y, vx, vy)
            nearest = min(nearest, time_to_collision(state, radius + Player.RADIUS))

        if nearest >= self.THREAT_HORIZON:
            return 0.0

        return 1.0 - nearest / self.THREAT_HORIZON

    def aim(self, world: GameWorld) -> tuple[float, int | None]:
        if not world.player.is_active:
            return 0.0, None

        targets = [(id(a), a.x, a.y, a.velocity_x, a.velocity_y) for a in world.asteroids]

        saucer = world.saucer
        if saucer is not None and saucer.is_alive:
            targets.append(
                (id(saucer), saucer.x, saucer.y, saucer.velocity_x, saucer.velocity_y)
            )

        best_time = NO_COLLISION
        best_angle = 0.0
        best_target: int | None = None

        for target, x, y, vx, vy in targets:
            shot = intercept(
                self._relative(world, x, y, vx, vy), Bullet.SPEED, Bullet.MAX_LIFETIME
            )

            if shot.reachable and shot.flight_time < best_time:
                best_time = shot.flight_time
                best_angle = shot.lead_angle
                best_target = target

        if best_target is None:
            return 0.0, None

        return (1.0 + math.cos(best_angle)) / 2.0, best_target

    def _misses(self, world: GameWorld) -> int:
        return world.shots_fired - world.accuracy_hits - len(world.bullets)

    def compute(
        self,
        world: GameWorld,
        action: ActionMap,
    ) -> float:
        rc = cfg.reward
        player = world.player

        reward = 0.0

        if player.is_active:
            reward += rc.survive_step

        reward += (world.score - self.previous_score) * rc.score_scale
        self.previous_score = world.score

        if world.player_lives < self.previous_lives:
            reward -= rc.life_lost
        self.previous_lives = world.player_lives

        if player.used_hyperspace_this_step:
            reward -= rc.hyperspace_cost

        misses = self._misses(world)
        reward -= (misses - self.previous_misses) * rc.missed_shot
        self.previous_misses = misses

        threat = self.threat(world)

        if self.previous_active and player.is_active:
            reward -= rc.danger_shaping * (threat - self.previous_threat)

        self.previous_threat = threat
        self.previous_active = player.is_active

        aim, target = self.aim(world)
        if target is not None and target == self.previous_target:
            reward += rc.aim_shaping * (aim - self.previous_aim)
        self.previous_aim = aim
        self.previous_target = target

        if world.is_done():
            reward -= rc.game_over

        return reward
