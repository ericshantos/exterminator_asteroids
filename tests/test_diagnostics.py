import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest  # noqa: E402

from configs import cfg  # noqa: E402
from entities import Asteroid, Bullet  # noqa: E402
from entities.ship.player.hyperspace_manager import HyperspaceManager  # noqa: E402
from env import AsteroidEnv  # noqa: E402
from env.action_space import Action  # noqa: E402


def make_env() -> AsteroidEnv:
    env = AsteroidEnv()
    env.reset(seed=0)
    env.world.asteroid_manager.clear()

    return env


def test_asteroid_death_is_recorded_by_cause() -> None:
    env = make_env()
    player = env.world.player
    env.world.asteroids.append(Asteroid(env.space, 3, player.x, player.y, 0.0, 0.0))

    *_, info = env.step(Action.NOTHING)

    assert info["deaths_asteroid"] == 1
    assert info["lives_lost"] == 1
    assert info["lives"] == cfg.game.starting_lives - 1


def test_asteroid_right_after_hyperspace_counts_as_landing() -> None:
    env = make_env()
    player = env.world.player
    player.frames_since_hyperspace = 0
    env.world.asteroids.append(Asteroid(env.space, 3, player.x, player.y, 0.0, 0.0))

    *_, info = env.step(Action.NOTHING)

    assert info["deaths_hyperspace_landing"] == 1
    assert info["deaths_asteroid"] == 0


def test_hyperspace_failure_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env()
    monkeypatch.setattr(
        env.world.player.hyperspace_manager, "should_explode", lambda count: True
    )

    env.step(Action.HYPERSPACE)

    for _ in range(HyperspaceManager.DURATION):
        *_, info = env.step(Action.NOTHING)

    assert info["hyperspace_jumps"] == 1
    assert info["deaths_hyperspace_failure"] == 1
    assert info["lives_lost"] == 1


def test_saucer_bullet_death_and_saucer_count() -> None:
    env = make_env()
    world = env.world
    player = world.player

    world.saucer_manager.spawn(world.score)
    assert world.saucer is not None
    world.saucer.fire(player.x, player.y + Bullet.SPEED, 0.0)

    *_, info = env.step(Action.NOTHING)

    assert info["saucers_spawned"] == 1
    assert info["deaths_saucer_bullet"] == 1
    assert info["points_per_life"] == world.score
