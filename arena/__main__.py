import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .match import MODEL_PATHS, format_report, load_model, play_match


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m arena",
        description="Joga uma partida até o game over com um modelo treinado.",
    )
    parser.add_argument("--algo", choices=MODEL_PATHS.keys(), default="dqn")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    model, path = load_model(args.algo, args.model)

    result = play_match(
        model,
        render=not args.no_render,
        seed=args.seed,
        deterministic=not args.stochastic,
        algo=args.algo,
        model_path=path,
    )

    print(format_report(result))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        print(f"\nResultado salvo em {args.output}")


if __name__ == "__main__":
    main()
