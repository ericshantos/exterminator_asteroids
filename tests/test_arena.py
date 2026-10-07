import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from arena import play_match  # noqa: E402
from configs import cfg  # noqa: E402
from env.action_space import Action  # noqa: E402


class SpinAndShoot:
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return np.array(Action.LEFT_SHOOT), None


def test_match_runs_until_game_over_past_training_limit() -> None:
    result = play_match(SpinAndShoot(), render=False, seed=3)

    assert result.end_reason == "game_over"
    assert result.lives_lost == cfg.game.starting_lives + result.extra_lives
    assert result.decisions == result.actions["LEFT_SHOOT"]
    assert result.hits <= result.shots_fired
    assert result.game_seconds == result.frames / cfg.screen.fps
