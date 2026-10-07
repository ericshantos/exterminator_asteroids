import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from configs import cfg  # noqa: E402
from entities import Asteroid  # noqa: E402
from env import AsteroidEnv, FrameSkip  # noqa: E402
from env.action_space import Action  # noqa: E402
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
