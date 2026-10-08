import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from configs import cfg  # noqa: E402
from entities import Asteroid, DeathCause  # noqa: E402
from env import AsteroidEnv, FrameSkip  # noqa: E402
from env.action_space import Action, ActionMap  # noqa: E402
from env.frame_skip import _RELEASED  # noqa: E402
from env.observation import Observation  # noqa: E402


def test_frame_skip_fires_every_step() -> None:
    env = FrameSkip(AsteroidEnv(), skip=4)
    env.reset(seed=0)

    shots = []
    for _ in range(4):
        *_, info = env.step(Action.SHOOT)
        shots.append(info["shots_fired"])

    assert shots == [1, 2, 3, 4]


def run_frame_by_frame(
    actions: list[Action], seed: int, max_frames: int
) -> list[tuple[np.ndarray, float, bool, bool]]:
    env = AsteroidEnv()
    env.max_episode_steps = max_frames
    obs, _ = env.reset(seed=seed)

    steps = [(obs, 0.0, False, False)]

    for action in actions:
        total = 0.0

        for frame in range(4):
            obs, reward, terminated, truncated, _ = env.step(
                action if frame == 0 else _RELEASED.get(action, action)
            )
            total += reward

            if terminated or truncated:
                break

        steps.append((obs, total, terminated, truncated))

        if terminated or truncated:
            break

    return steps


def test_frame_skip_observation_matches_frame_by_frame() -> None:
    rng = np.random.default_rng(0)
    actions = [Action(int(a)) for a in rng.integers(0, len(Action), size=200)]

    # Truncado no meio de um bloco de 4 frames.
    max_frames = 4 * 150 + 2

    expected = run_frame_by_frame(actions, seed=5, max_frames=max_frames)

    env = FrameSkip(AsteroidEnv(), skip=4)
    env.unwrapped.max_episode_steps = max_frames  # type: ignore[attr-defined]
    obs, _ = env.reset(seed=5)

    steps = [(obs, 0.0, False, False)]

    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(action)
        steps.append((obs, reward, terminated, truncated))

        if terminated or truncated:
            break

    assert len(steps) == len(expected)
    assert steps[-1][2] or steps[-1][3]

    for (obs, reward, terminated, truncated), (e_obs, e_rew, e_term, e_trunc) in zip(
        steps, expected
    ):
        assert np.array_equal(obs, e_obs)
        assert np.isclose(reward, e_rew)
        assert (terminated, truncated) == (e_term, e_trunc)


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


def radar_with(size: int, dx: float, dy: float, vx: float, vy: float) -> np.ndarray:
    env = AsteroidEnv()
    env.reset(seed=0)

    world = env.world
    world.asteroid_manager.clear()

    player = world.player
    player.angle = 0.0
    player.velocity_x = player.velocity_y = 0.0

    world.asteroids.append(
        Asteroid(env.space, size, player.x + dx, player.y + dy, vx, vy)
    )

    return env.observation.build(world)[-Observation.RADAR_FEATURES :]


def test_radar_asteroid_from_the_right() -> None:
    radar = radar_with(3, 150.0, 0.0, -3.0, 0.0)
    sectors, course, best_angle, best_value = radar[:16], *radar[16:]

    assert np.all(sectors[[0, 1]] == 1.0)
    assert np.all(sectors[2:10] < 0.3)
    assert np.all(sectors[10:] == 1.0)
    assert np.isclose(course, 34.0 / 120.0)
    assert best_angle == 0.0 and best_value == 1.0


def test_radar_asteroid_head_on() -> None:
    radar = radar_with(2, 0.0, -150.0, 0.0, 3.0)
    sectors, course, best_angle, best_value = radar[:16], *radar[16:]

    assert sectors[0] < 0.2
    assert sectors[1] < 0.3 and sectors[15] < 0.3
    assert sectors[8] < 0.4
    assert np.all(sectors[2:8] == 1.0) and np.all(sectors[9:15] == 1.0)
    assert np.isclose(course, (122.0 / 3.0) / 120.0)
    assert np.isclose(best_angle, 0.25) and best_value == 1.0


def test_radar_receding_asteroid_is_all_clear() -> None:
    radar = radar_with(3, 150.0, 0.0, 3.0, 0.0)

    assert np.all(radar[:17] == 1.0)
    assert radar[17] == 0.0 and radar[18] == 1.0


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
