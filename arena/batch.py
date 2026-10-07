import math
import multiprocessing
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .match import DEATH_LABELS, Algo, MatchResult, Policy, load_model, play_match

EVALUATION_SEED: int = 1000

Z_95: float = 1.96

METRICS: dict[str, str] = {
    "points_per_life": "Pontos por vida",
    "score": "Pontuação",
    "game_seconds": "Duração (s)",
    "wave_reached": "Onda alcançada",
    "lives_lost": "Vidas perdidas",
    "saucer_kill_rate": "Discos destruídos",
    "accuracy": "Precisão",
    "hyperspace_jumps": "Hiperespaços",
    "points_per_minute": "Pontos por minuto",
}

PERCENT_METRICS: frozenset[str] = frozenset({"saucer_kill_rate", "accuracy"})


@dataclass
class Stat:
    mean: float
    std: float
    ci95: float
    minimum: float
    maximum: float


@dataclass
class BatchResult:
    algo: str
    model_path: str
    first_seed: int
    requested: int
    deterministic: bool
    wall_seconds: float
    pooled_points_per_life: float
    stats: dict[str, Stat] = field(default_factory=dict)
    deaths: dict[str, int] = field(default_factory=dict)
    matches: list[MatchResult] = field(default_factory=list)


_worker_model: Policy | None = None
_worker_algo: str = ""
_worker_path: str = ""
_worker_deterministic: bool = True


def _init_worker(algo: Algo, path: str, deterministic: bool) -> None:
    global _worker_model, _worker_algo, _worker_path, _worker_deterministic

    import torch

    torch.set_num_threads(1)

    _worker_model, _ = load_model(algo, Path(path))
    _worker_algo = algo
    _worker_path = path
    _worker_deterministic = deterministic


def _play_in_worker(seed: int) -> MatchResult:
    assert _worker_model is not None

    return play_match(
        _worker_model,
        render=False,
        seed=seed,
        deterministic=_worker_deterministic,
        algo=_worker_algo,
        model_path=_worker_path,
    )


def _stat(values: list[float]) -> Stat:
    std = statistics.stdev(values) if len(values) > 1 else 0.0

    return Stat(
        mean=statistics.fmean(values),
        std=std,
        ci95=Z_95 * std / math.sqrt(len(values)),
        minimum=min(values),
        maximum=max(values),
    )


def summarize(
    matches: list[MatchResult],
    first_seed: int,
    requested: int,
    deterministic: bool = True,
    wall_seconds: float = 0.0,
) -> BatchResult:
    if not matches:
        raise ValueError("No finished matches to summarize.")

    deaths = {
        cause: sum(m.deaths.get(cause, 0) for m in matches) for cause in DEATH_LABELS
    }

    total_lives = sum(m.lives_lost for m in matches)

    return BatchResult(
        algo=matches[0].algo,
        model_path=matches[0].model_path,
        first_seed=first_seed,
        requested=requested,
        deterministic=deterministic,
        wall_seconds=wall_seconds,
        pooled_points_per_life=sum(m.score for m in matches) / max(total_lives, 1),
        stats={
            key: _stat([float(getattr(m, key)) for m in matches]) for key in METRICS
        },
        deaths=deaths,
        matches=sorted(matches, key=lambda m: m.seed or 0),
    )


def play_batch(
    algo: Algo,
    model_path: Path | None,
    matches: int,
    first_seed: int = EVALUATION_SEED,
    deterministic: bool = True,
    render: bool = False,
    workers: int = 1,
    on_result: Callable[[MatchResult, int], None] | None = None,
) -> BatchResult:
    model, path = load_model(algo, model_path)

    seeds = range(first_seed, first_seed + matches)
    finished: list[MatchResult] = []
    started = time.perf_counter()

    def collect(result: MatchResult) -> bool:
        if result.end_reason != "game_over":
            return False

        finished.append(result)

        if on_result is not None:
            on_result(result, len(finished))

        return True

    if workers <= 1:
        for seed in seeds:
            result = play_match(
                model,
                render=render,
                seed=seed,
                deterministic=deterministic,
                algo=algo,
                model_path=path,
            )

            if not collect(result):
                break
    else:
        context = multiprocessing.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(algo, str(path), deterministic),
        ) as pool:
            futures = [pool.submit(_play_in_worker, seed) for seed in seeds]

            try:
                for future in as_completed(futures):
                    collect(future.result())
            except KeyboardInterrupt:
                pool.shutdown(wait=False, cancel_futures=True)

    return summarize(
        finished,
        first_seed,
        matches,
        deterministic,
        time.perf_counter() - started,
    )


def _format_value(key: str, value: float) -> str:
    if key in PERCENT_METRICS:
        return f"{value:.1%}"

    return f"{value:.1f}"


def format_batch_report(batch: BatchResult) -> str:
    played = len(batch.matches)
    last_seed = batch.first_seed + batch.requested - 1

    lines = [
        "",
        "=== ARENA: AVALIAÇÃO EM LOTE ===",
        f"Modelo:     {batch.algo.upper()} ({batch.model_path})",
        f"Partidas:   {played} de {batch.requested} "
        f"(seeds {batch.first_seed} a {last_seed}), "
        f"{'determinístico' if batch.deterministic else 'estocástico'}",
        f"Tempo real: {batch.wall_seconds:.1f} s",
        "",
        f"{'Métrica':<20} {'Média':>10} {'± IC 95%':>10} {'Desvio':>10} "
        f"{'Mín':>10} {'Máx':>10}",
    ]

    for key, label in METRICS.items():
        s = batch.stats[key]
        values = (s.mean, s.ci95, s.std, s.minimum, s.maximum)
        lines.append(
            f"{label:<20} " + " ".join(f"{_format_value(key, v):>10}" for v in values)
        )

    total_deaths = sum(batch.deaths.values())

    lines += [
        "",
        f"Pontos por vida (total): {batch.pooled_points_per_life:.0f} "
        "(soma dos pontos ÷ soma das vidas perdidas)",
        "",
        f"Mortes por causa ({total_deaths} no total):",
    ]

    lines += [
        f"  {DEATH_LABELS[cause]:<27} {n:>5}  {n / max(total_deaths, 1):6.1%}"
        for cause, n in sorted(batch.deaths.items(), key=lambda i: -i[1])
    ]

    if played < batch.requested:
        lines += [
            "",
            f"Atenção: só {played} das {batch.requested} partidas terminaram em "
            "game over. As demais foram interrompidas e ficaram de fora.",
        ]

    return "\n".join(lines)
