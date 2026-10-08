import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from stable_baselines3 import DQN, PPO

from configs import cfg
from env.action_space import Action
from env.frame_skip import FrameSkip
from env.observation import Observation

from .arena_env import ArenaEnv

Algo = Literal["dqn", "ppo"]
EndReason = Literal[
    "game_over", "time_limit", "score_limit", "window_closed", "interrupted"
]

DEATH_LABELS: dict[str, str] = {
    "saucer_bullet": "Tiro do disco",
    "asteroid": "Asteroide",
    "hyperspace_failure": "Falha do hiperespaço",
    "hyperspace_landing": "Asteroide após hiperespaço",
    "saucer_collision": "Colisão com o disco",
}

MODEL_PATHS: dict[str, Path] = {
    "dqn": Path("models/exterminator_dqn.zip"),
    "ppo": Path("models/exterminator_ppo.zip"),
}


class Policy(Protocol):
    def predict(
        self, observation: np.ndarray, deterministic: bool = ...
    ) -> tuple[np.ndarray, object]: ...


@dataclass
class MatchResult:
    algo: str
    model_path: str
    seed: int | None
    end_reason: EndReason
    score: int
    wave_reached: int
    frames: int
    game_seconds: float
    wall_seconds: float
    decisions: int
    asteroids_destroyed: int
    saucers_destroyed: int
    shots_fired: int
    hits: int
    accuracy: float
    lives_lost: int
    extra_lives: int
    peak_lives: int
    hyperspace_jumps: int
    saucers_spawned: int
    saucer_kill_rate: float
    points_per_minute: float
    points_per_life: float
    seconds_per_life: float
    deaths: dict[str, int] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)


def load_model(algo: Algo, path: Path | None = None) -> tuple[Policy, Path]:
    path = path or MODEL_PATHS[algo]

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model = (
        PPO.load(path, device="cpu") if algo == "ppo" else DQN.load(path, device="cpu")
    )

    shape = model.observation_space.shape

    if shape != (Observation.SIZE,):
        raise ValueError(
            f"{path} expects observations of shape {shape}, but the environment "
            f"produces ({Observation.SIZE},). Retrain the model."
        )

    return model, path


def play_match(
    model: Policy,
    render: bool = True,
    seed: int | None = None,
    deterministic: bool = True,
    algo: str = "",
    model_path: Path | str = "",
    max_minutes: float | None = None,
    max_score: int | None = None,
) -> MatchResult:
    arena = ArenaEnv(render=render)
    env = FrameSkip(arena, cfg.rl.frame_skip)

    actions: Counter[Action] = Counter()
    end_reason: EndReason = "game_over"
    started = time.perf_counter()

    max_frames = (
        None if max_minutes is None else round(max_minutes * 60 * cfg.screen.fps)
    )

    try:
        obs, _ = env.reset(seed=seed)
        terminated = False

        while not terminated:
            if max_frames is not None and arena.world.frame_count >= max_frames:
                end_reason = "time_limit"
                break

            if max_score is not None and arena.world.score >= max_score:
                end_reason = "score_limit"
                break

            prediction, _ = model.predict(obs, deterministic=deterministic)
            action = Action(int(prediction))
            actions[action] += 1

            obs, _, terminated, _, _ = env.step(action)

        if arena.closed_by_user:
            end_reason = "window_closed"
    except KeyboardInterrupt:
        end_reason = "interrupted"
    finally:
        env.close()

    return _summarize(
        arena,
        actions,
        end_reason,
        time.perf_counter() - started,
        seed,
        algo,
        str(model_path),
    )


def _summarize(
    arena: ArenaEnv,
    actions: Counter[Action],
    end_reason: EndReason,
    wall_seconds: float,
    seed: int | None,
    algo: str,
    model_path: str,
) -> MatchResult:
    world = arena.world
    game_seconds = world.frame_count / cfg.screen.fps
    lives_used = max(world.lives_lost, 1)

    return MatchResult(
        algo=algo,
        model_path=model_path,
        seed=seed,
        end_reason=end_reason,
        score=world.score,
        wave_reached=world.wave,
        frames=world.frame_count,
        game_seconds=game_seconds,
        wall_seconds=wall_seconds,
        decisions=sum(actions.values()),
        asteroids_destroyed=world.asteroids_destroyed,
        saucers_destroyed=world.saucers_destroyed,
        shots_fired=world.shots_fired,
        hits=world.accuracy_hits,
        accuracy=world.accuracy,
        lives_lost=world.lives_lost,
        extra_lives=arena.extra_lives,
        peak_lives=arena.peak_lives,
        hyperspace_jumps=world.hyperspace_jumps,
        saucers_spawned=world.saucers_spawned,
        saucer_kill_rate=(
            world.saucers_destroyed / world.saucers_spawned
            if world.saucers_spawned
            else 0.0
        ),
        points_per_life=world.points_per_life,
        deaths=world.deaths,
        points_per_minute=world.score / (game_seconds / 60) if game_seconds else 0.0,
        seconds_per_life=game_seconds / lives_used,
        actions={a.name: actions[a] for a in Action},
    )


def format_report(result: MatchResult) -> str:
    total = max(result.decisions, 1)
    used = sorted(
        ((name, n) for name, n in result.actions.items() if n),
        key=lambda item: item[1],
        reverse=True,
    )

    minutes, seconds = divmod(result.game_seconds, 60)

    lines = [
        "",
        "=== ARENA: FIM DE PARTIDA ===",
        f"Modelo:               {result.algo.upper()} ({result.model_path})",
        f"Seed:                 {result.seed}",
        f"Fim:                  {result.end_reason}",
        "",
        f"Pontuação:            {result.score}",
        f"Onda alcançada:       {result.wave_reached}",
        f"Tempo de jogo:        {int(minutes)}m{seconds:04.1f}s "
        f"({result.frames} frames)",
        f"Tempo real:           {result.wall_seconds:.1f} s",
        f"Pontos por minuto:    {result.points_per_minute:.1f}",
        f"Pontos por vida:      {result.points_per_life:.0f}",
        "",
        f"Asteroides destruídos: {result.asteroids_destroyed}",
        f"Discos destruídos:    {result.saucers_destroyed} de "
        f"{result.saucers_spawned} ({result.saucer_kill_rate:.0%})",
        f"Tiros disparados:     {result.shots_fired}",
        f"Acertos:              {result.hits}",
        f"Precisão:             {result.accuracy:.2%}",
        "",
        f"Vidas perdidas:       {result.lives_lost}",
        f"Vidas extras:         {result.extra_lives}",
        f"Máximo de vidas:      {result.peak_lives}",
        f"Tempo médio por vida: {result.seconds_per_life:.1f} s",
        f"Hiperespaços:         {result.hyperspace_jumps}",
        "",
        "Mortes por causa:",
    ]

    lost = max(result.lives_lost, 1)

    lines += [
        f"  {DEATH_LABELS.get(cause, cause):<27} {n:>4}  {n / lost:6.1%}"
        for cause, n in sorted(result.deaths.items(), key=lambda i: -i[1])
    ]

    lines += ["", f"Decisões do agente:   {result.decisions}"]

    lines += [f"  {name:<20} {n:>7}  {n / total:6.1%}" for name, n in used]

    return "\n".join(lines)
