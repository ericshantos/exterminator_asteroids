import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .batch import EVALUATION_SEED, format_batch_report, play_batch
from .match import MODEL_PATHS, MatchResult, format_report, load_model, play_match


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nResultado salvo em {path}")


def _progress(total: int) -> Any:
    def report(result: MatchResult, done: int) -> None:
        print(
            f"[{done:>{len(str(total))}}/{total}] seed {result.seed}: "
            f"{result.score} pts, {result.lives_lost} vidas perdidas, "
            f"{result.points_per_life:.0f} pts/vida",
            flush=True,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m arena",
        description="Joga partidas até o game over com um modelo treinado.",
    )
    parser.add_argument("--algo", choices=MODEL_PATHS.keys(), default="dqn")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--matches", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.matches < 1:
        parser.error("--matches must be at least 1")

    render = not args.no_render

    if args.matches == 1:
        model, path = load_model(args.algo, args.model)

        result = play_match(
            model,
            render=render,
            seed=args.seed,
            deterministic=not args.stochastic,
            algo=args.algo,
            model_path=path,
        )

        print(format_report(result))

        if args.output is not None:
            _save(args.output, asdict(result))

        return

    workers = args.workers or (1 if render else min(args.matches, os.cpu_count() or 1))

    if render and workers > 1:
        parser.error("--workers > 1 requires --no-render")

    first_seed = EVALUATION_SEED if args.seed is None else args.seed

    print(
        f"Jogando {args.matches} partidas (seeds {first_seed} a "
        f"{first_seed + args.matches - 1}) com {workers} processo(s)...",
        flush=True,
    )

    batch = play_batch(
        args.algo,
        args.model,
        args.matches,
        first_seed=first_seed,
        deterministic=not args.stochastic,
        render=render,
        workers=workers,
        on_result=_progress(args.matches),
    )

    print(format_batch_report(batch))

    if args.output is not None:
        _save(args.output, asdict(batch))


if __name__ == "__main__":
    main()
