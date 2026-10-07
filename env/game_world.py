from configs import cfg
from entities import (
    Asteroid,
    AsteroidManager,
    Bullet,
    DeathCause,
    Player,
    Saucer,
    SaucerManager,
)

from .action_space import ActionMap
from .collision_manager import CollisionManager
from .respawn_manager import RespawnManager
from .score_manager import ScoreManager
from .toroidal_space import ToroidalSpace


class GameWorld:
    WAVE_DELAY: int = 90

    WAVE_INCREMENT: int = 2

    def __init__(self, space: ToroidalSpace) -> None:
        self.space = space

        self.frame_count: int
        self.done: bool

        self.player: Player
        self.asteroid_manager: AsteroidManager
        self.score_manager: ScoreManager
        self.collision_manager: CollisionManager

        self.initial_asteroids: int
        self.wave: int
        self.wave_timer: int

        self.saucer_spawn_timer: int

        self.saucer_manager: SaucerManager

        self.reset()

    def reset(self) -> None:
        self.frame_count = 0

        self.wave = 1
        self.wave_timer = 0

        self.saucer_spawn_timer = 0

        self.initial_asteroids = cfg.game.initial_asteroids

        self.done = False

        self.player = Player(*self.center_world, self.space)

        self.saucer_manager = SaucerManager(self.space)

        self.asteroid_manager = AsteroidManager(self.space)

        self.score_manager = ScoreManager(self.player)

        self.collision_manager = CollisionManager(
            self.player, self.asteroid_manager, self.score_manager, self.space
        )

        self.asteroid_manager.spawn_wave(
            self.initial_asteroids, self.player.x, self.player.y
        )

    def _wave_size(self, wave: int) -> int:
        return min(
            self.initial_asteroids + (wave - 1) * self.WAVE_INCREMENT,
            cfg.game.max_asteroids,
        )

    def _next_wave(self) -> None:
        self.wave += 1

        self.asteroid_manager.spawn_wave(
            self._wave_size(self.wave), self.player.x, self.player.y
        )

    def _update_saucer_spawn(self) -> None:
        if self.saucer_manager.saucer is not None:
            return

        self.saucer_spawn_timer += 1

        if self.saucer_spawn_timer > self.saucer_manager.spawn_interval(self.score):
            self.saucer_manager.spawn(self.score)

            self.saucer_spawn_timer = 0

    def _update_wave(self) -> None:
        if not self.asteroid_manager.is_empty:
            return

        if self.wave_timer == 0:
            self.wave_timer = self.WAVE_DELAY
            return

        self.wave_timer -= 1

        if self.wave_timer == 0:
            self._next_wave()

    def update(self, action: ActionMap) -> None:
        if self.done:
            return

        self.frame_count += 1

        self._update_saucer_spawn()

        self.player.update(action, self.asteroid_manager.count)

        self.asteroid_manager.update()

        self.saucer_manager.update(self.player.x, self.player.y)

        self.collision_manager.update(self.saucer)

        self._handle_respawn()

        self._update_wave()

        if self.is_game_over():
            self.done = True

    def _handle_respawn(self) -> None:
        if self.player.is_alive:
            return

        if self.is_game_over():
            return

        if self.player.respawn_timer < self.player.RESPAWN_DELAY:
            return

        if RespawnManager.can_respawn(
            self.space,
            self.player.start_x,
            self.player.start_y,
            self.asteroids,
            self.saucer,
        ):
            self.player.respawn()

    @property
    def player_alive(self) -> bool:
        return self.player.is_alive

    @property
    def player_lives(self) -> int:
        return self.player.lives

    @property
    def score(self) -> int:
        return self.score_manager.score

    def is_done(self) -> bool:
        return self.done

    def is_game_over(self) -> bool:
        return self.player.lives <= 0

    @property
    def bullets(self) -> list[Bullet]:
        return self.player.bullet_manager.bullets

    @property
    def asteroids(self) -> list[Asteroid]:
        return self.asteroid_manager.asteroids

    @property
    def asteroids_destroyed(self) -> int:
        return self.collision_manager.asteroids_destroyed

    @property
    def saucers_destroyed(self) -> int:
        return self.collision_manager.saucers_destroyed

    @property
    def accuracy_hits(self) -> int:
        return self.collision_manager.accuracy_hits

    @property
    def shots_fired(self) -> int:
        return self.player.shots_fired

    @property
    def saucer(self) -> Saucer | None:
        return self.saucer_manager.saucer

    @property
    def accuracy(self) -> float:
        if self.shots_fired == 0:
            return 0.0

        return self.accuracy_hits / self.shots_fired

    @property
    def lives_lost(self) -> int:
        return self.player.lives_lost

    @property
    def deaths(self) -> dict[str, int]:
        return {cause.value: self.player.deaths[cause] for cause in DeathCause}

    @property
    def saucers_spawned(self) -> int:
        return self.saucer_manager.spawned

    @property
    def hyperspace_jumps(self) -> int:
        return self.player.hyperspace_jumps

    @property
    def points_per_life(self) -> float:
        return self.score / max(self.lives_lost, 1)

    def diagnostics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "lives_lost": self.lives_lost,
            "points_per_life": self.points_per_life,
            "saucers_spawned": self.saucers_spawned,
            "saucers_destroyed": self.saucers_destroyed,
            "hyperspace_jumps": self.hyperspace_jumps,
        }

        for cause, count in self.deaths.items():
            metrics[f"deaths_{cause}"] = count

        return metrics

    @property
    def center_world(self) -> tuple[int, int]:
        return self.space.width // 2, self.space.height // 2
