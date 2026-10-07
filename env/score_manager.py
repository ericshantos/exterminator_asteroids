from configs import cfg
from entities import Player


class ScoreManager:
    DISPLAY_ROLLOVER: int = 100000

    def __init__(self, player: Player) -> None:
        self.player = player

        self.extra_life_score: int = cfg.game.extra_life_score

        self.score: int
        self.next_extra_life: int

        self.reset()

    def __str__(self) -> str:
        return str(self.score)

    def add(self, points: int) -> None:
        if self.player.lives == 0 and not self.player.is_alive:
            return

        self.score += points

        while self.score >= self.next_extra_life:
            self.player.gain_life()
            self.next_extra_life += self.extra_life_score

    def get_score(self) -> str:
        displayed = self.score % self.DISPLAY_ROLLOVER

        if displayed == 0:
            return "00"

        return str(displayed)

    def reset(self) -> None:
        self.score = 0
        self.next_extra_life = self.extra_life_score
