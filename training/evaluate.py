import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO

from configs import cfg
from env import make_env
import numpy as np


MODEL_PATHS = {
    "dqn": Path("models/exterminator_dqn.zip"),
    "ppo": Path("models/exterminator_ppo.zip"),
}
EPISODES = 20


def evaluate(
    algo: str = "dqn",
    model_path: Path | None = None,
    episodes: int = EPISODES,
    render: bool = True,
    seed: int | None = None,
):
    env = make_env(render_mode="human" if render else None)

    path = model_path or MODEL_PATHS[algo]

    if algo == "ppo":
        model = PPO.load(path, device="cpu")
    else:
        model = DQN.load(path)

    scores = []
    survival_times = []
    accuracies = []
    asteroids_destroyed = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode if seed is not None else None)

        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)

        scores.append(info.get("score", 0))
        survival_times.append(info.get("frame_count", 0) / cfg.screen.fps)
        accuracies.append(info.get("accuracy", 0))
        asteroids_destroyed.append(
            info.get("asteroid_destroyed", 0)
        )

        print(
            f"Episode {episode + 1:02d} | "
            f"Score={scores[-1]} | "
            f"Asteroids={asteroids_destroyed[-1]} | "
            f"Accuracy={accuracies[-1]:.2%}"
        )

    print("\n=== RESULTS ===")

    print(f"Episodes: {episodes}")

    print(
        f"Average score: "
        f"{np.mean(scores):.2f}"
    )

    print(
        f"Average survival time: "
        f"{np.mean(survival_times):.2f} s"
    )

    print(
        f"Average accuracy: "
        f"{np.mean(accuracies):.2%}"
    )

    print(
        f"Average asteroids destroyed: "
        f"{np.mean(asteroids_destroyed):.2f}"
    )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=MODEL_PATHS.keys(), default="dqn")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")

    args = parser.parse_args()

    evaluate(args.algo, args.model, args.episodes, not args.no_render, args.seed)