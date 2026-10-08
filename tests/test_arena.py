import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from arena import format_batch_report, play_match, summarize  # noqa: E402
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
    assert sum(result.deaths.values()) == result.lives_lost
    assert result.decisions == result.actions["LEFT_SHOOT"]
    assert result.hits <= result.shots_fired
    assert result.game_seconds == result.frames / cfg.screen.fps


def test_batch_summary_aggregates_matches() -> None:
    results = [play_match(SpinAndShoot(), render=False, seed=s) for s in (3, 4)]
    batch = summarize(results, first_seed=3, requested=2)

    scores = [r.score for r in results]
    lives = sum(r.lives_lost for r in results)

    assert batch.stats["score"].mean == sum(scores) / 2
    assert batch.stats["score"].minimum == min(scores)
    assert batch.stats["score"].maximum == max(scores)
    assert sum(batch.deaths.values()) == lives
    assert batch.pooled_points_per_life == sum(scores) / lives
    assert [m.seed for m in batch.matches] == [3, 4]
    assert "AVALIAÇÃO EM LOTE" in format_batch_report(batch)


def test_match_stops_at_time_limit() -> None:
    result = play_match(SpinAndShoot(), render=False, seed=3, max_minutes=0.05)
    limit = round(0.05 * 60 * cfg.screen.fps)

    assert result.end_reason == "time_limit"
    assert limit <= result.frames < limit + cfg.rl.frame_skip

    batch = summarize([result], first_seed=3, requested=1, max_minutes=0.05)

    assert "1 de 1 partidas chegaram ao limite" in format_batch_report(batch)


def test_match_stops_at_score_limit() -> None:
    result = play_match(SpinAndShoot(), render=False, seed=3, max_score=500)

    assert result.end_reason == "score_limit"
    assert result.score >= 500

    batch = summarize([result], first_seed=3, requested=1, max_score=500)

    assert "1 de 1 partidas chegaram ao limite" in format_batch_report(batch)
