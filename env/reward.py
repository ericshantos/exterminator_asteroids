from configs import cfg
import numpy as np

from .action_space import ActionMap
from .danger_model import DangerModel
from .game_world import GameWorld
from .toroidal_space import ToroidalSpace


class RewardFunction:
    def __init__(self, space: ToroidalSpace) -> None:
        self.space = space
        self.danger_model = DangerModel(space)

        self.previous_score = 0
        self.previous_lives = cfg.game.starting_lives
        self.previous_risk = 0.0

    def reset(self) -> None:
        self.previous_score = 0
        self.previous_lives = cfg.game.starting_lives
        self.previous_risk = 0.0

    def _aim_target(self, world: GameWorld) -> tuple[float, float] | None:
        saucer = world.saucer

        if saucer is not None and saucer.is_alive:
            return saucer.x, saucer.y

        if not world.asteroids:
            return None

        player = world.player

        nearest = min(
            world.asteroids,
            key=lambda a: self.space.distance(player.x, player.y, a.x, a.y),
        )

        return nearest.x, nearest.y

    def compute(
        self,
        world: GameWorld,
        action: ActionMap,
    ) -> float:

        reward = cfg.reward.survive_step

        # -------------------------
        # SCORE (SPARSE SIGNAL)
        # -------------------------
        current_score = world.score
        score_delta = current_score - self.previous_score
        reward += score_delta * 0.01
        self.previous_score = current_score

        # -------------------------
        # LIVES (STRONG NEGATIVE SIGNAL)
        # -------------------------
        current_lives = world.player_lives
        if current_lives < self.previous_lives:
            reward -= 3.0
        self.previous_lives = current_lives

        # -------------------------
        # ACTION COST (SMALL REGULARIZATION)
        # -------------------------
        if action.rotate_left:
            reward -= 0.001

        if action.rotate_right:
            reward -= 0.001

        if action.thrust:
            reward -= 0.002

        if action.shoot:
            reward -= 0.001  # ↓ reduzido (não pode punir combate forte)

        if action.hyperspace:
            reward -= 0.05

        # -------------------------
        # DANGER SHAPING
        # -------------------------
        analysis = self.danger_model.metrics(
            world.player,
            world.asteroids,
        )

        current_risk = analysis.top_risk

        reward -= 0.03 * current_risk  # ↓ reduzido para não dominar combate
        reward += 0.02 * (self.previous_risk - current_risk)

        self.previous_risk = current_risk

        # -------------------------
        # AIM REWARD (CRÍTICO PARA ATIRADOR)
        # -------------------------
        player = world.player

        target = self._aim_target(world)

        alignment = 0.0

        if target is not None and player.is_active:
            dx, dy = self.space.relative_vector(
                player.x, player.y, target[0], target[1]
            )

            heading_x, heading_y = player.heading

            distance = float(np.hypot(dx, dy)) + 1e-6

            alignment = (dx * heading_x + dy * heading_y) / distance

            # reward contínuo de mira (tracking)
            reward += 0.01 * alignment

        # -------------------------
        # SHOOT REWARD (TRANSFORMA EM POLÍTICA DE PRECISÃO)
        # -------------------------
        if player.fired_this_step:
            reward += 0.5 * alignment
            reward -= 0.001  # custo mínimo para não spammar

        # -------------------------
        # TERMINAL
        # -------------------------
        if world.is_done():
            reward -= 10.0

        return reward