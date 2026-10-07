import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from configs import cfg  # noqa: E402
from entities import Asteroid, DeathCause  # noqa: E402
from env import AsteroidEnv, FrameSkip  # noqa: E402
from env.action_space import Action, ActionMap  # noqa: E402
from env.observation import Observation  # noqa: E402


def test_frame_skip_fires_every_step() -> None:
    env = FrameSkip(AsteroidEnv(), skip=4)
    env.reset(seed=0)

    shots = []
    for _ in range(4):
        *_, info = env.step(Action.SHOOT)
        shots.append(info["shots_fired"])

    assert shots == [1, 2, 3, 4]


def test_observation_shape_and_bounds() -> None:
    env = AsteroidEnv()
    obs, _ = env.reset(seed=1)

    for _ in range(2000):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs.shape == (Observation.SIZE,)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        if terminated or truncated:
            obs, _ = env.reset()


def test_seed_reproduces_episode() -> None:
    a, b = AsteroidEnv(), AsteroidEnv()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)

    for _ in range(300):
        oa, *_ = a.step(Action.SHOOT)
        ob, *_ = b.step(Action.SHOOT)

    assert np.array_equal(oa, ob)


def test_missed_shot_is_penalised() -> None:
    env = AsteroidEnv()
    env.reset(seed=0)
    env.world.asteroid_manager.clear()

    env.step(Action.SHOOT)

    rewards = [env.step(Action.NOTHING)[1] for _ in range(70)]

    expected = -cfg.reward.missed_shot + cfg.reward.survive_step
    assert any(np.isclose(r, expected) for r in rewards)


def test_threat_rises_when_asteroid_approaches() -> None:
    env = AsteroidEnv()
    env.reset(seed=0)

    world = env.world
    world.asteroid_manager.clear()

    player = world.player
    world.asteroids.append(
        Asteroid(env.space, 3, player.x + 150, player.y, -3.0, 0.0)
    )

    assert env.reward_system.threat(world) > 0.0

    world.asteroids[0].velocity_x = 3.0
    assert env.reward_system.threat(world) == 0.0


def test_aim_is_full_when_target_is_dead_ahead() -> None:
    env = AsteroidEnv()
    env.reset(seed=0)

    world = env.world
    world.asteroid_manager.clear()

    player = world.player
    world.asteroids.append(Asteroid(env.space, 3, player.x, player.y - 200, 0.0, 0.0))

    aim, target = env.reward_system.aim(world)
    assert target is not None
    assert np.isclose(aim, 1.0)

    player.angle = 180.0
    aim, _ = env.reward_system.aim(world)
    assert aim < 0.05


def env_under_threat() -> AsteroidEnv:
    env = AsteroidEnv()
    env.reset(seed=0)

    world = env.world
    world.asteroid_manager.clear()

    player = world.player
    world.asteroids.append(Asteroid(env.space, 3, player.x + 150, player.y, -3.0, 0.0))

    env.reward_system.compute(world, ActionMap())
    assert env.reward_system.threat(world) > 0.5

    return env


def enter_hyperspace(env: AsteroidEnv) -> float:
    hyperspace = ActionMap(hyperspace=True)
    env.world.player.update(hyperspace)

    return env.reward_system.compute(env.world, hyperspace)


def leave_hyperspace(env: AsteroidEnv) -> float:
    player = env.world.player
    player.update(ActionMap())
    player.hyperspace_manager.reset()

    return env.reward_system.compute(env.world, ActionMap())


def test_entering_hyperspace_gives_no_threat_credit() -> None:
    env = env_under_threat()

    reward = enter_hyperspace(env)

    assert env.world.player.in_hyperspace
    assert reward == pytest.approx(-cfg.reward.hyperspace_cost)


def test_reappearing_in_danger_is_not_charged() -> None:
    env = env_under_threat()
    enter_hyperspace(env)

    reward = leave_hyperspace(env)

    assert env.world.player.is_active
    assert env.reward_system.threat(env.world) > 0.5
    assert reward == pytest.approx(cfg.reward.survive_step)


def test_threat_shaping_resumes_after_reactivation() -> None:
    env = env_under_threat()
    enter_hyperspace(env)
    leave_hyperspace(env)

    world = env.world
    before = env.reward_system.threat(world)
    world.asteroids[0].x -= 30
    after = env.reward_system.threat(world)

    reward = env.reward_system.compute(world, ActionMap())

    expected = cfg.reward.survive_step - cfg.reward.danger_shaping * (after - before)
    assert after > before
    assert reward == pytest.approx(expected)


def test_death_costs_the_full_life_without_threat_credit() -> None:
    env = env_under_threat()
    world = env.world

    world.player.die(DeathCause.ASTEROID)
    reward = env.reward_system.compute(world, ActionMap())

    assert world.player_lives == cfg.game.starting_lives - 1
    assert reward == pytest.approx(-cfg.reward.life_lost)
