from entities import Asteroid, AsteroidManager, Bullet, BulletManager, Player, Saucer

from .score_manager import ScoreManager
from .toroidal_space import ToroidalSpace


class CollisionManager:
    def __init__(
        self,
        player: Player,
        asteroid_manager: AsteroidManager,
        score_manager: ScoreManager,
        space: ToroidalSpace,
    ) -> None:
        self.player = player
        self.asteroid_manager = asteroid_manager
        self.score_manager = score_manager
        self.space = space

        self.reset()

    def reset(self) -> None:
        self.asteroids_destroyed = 0
        self.saucers_destroyed = 0
        self.accuracy_hits = 0

    def _touching(
        self, ax: float, ay: float, a_radius: float, bx: float, by: float, b_radius: float
    ) -> bool:
        return self.space.distance(ax, ay, bx, by) <= a_radius + b_radius

    def _bullets_vs_asteroids(self, bullet_manager: BulletManager) -> list[Asteroid]:
        hit: list[Asteroid] = []
        spent: list[Bullet] = []

        for bullet in list(bullet_manager.bullets):
            for asteroid in self.asteroid_manager.asteroids:
                if asteroid in hit:
                    continue

                if self._touching(
                    asteroid.x, asteroid.y, asteroid.radius,
                    bullet.x, bullet.y, bullet.RADIUS,
                ):
                    hit.append(asteroid)
                    spent.append(bullet)
                    break

        for bullet in spent:
            bullet_manager.remove(bullet)

        for asteroid in hit:
            self.asteroid_manager.remove(asteroid)

        return hit

    def check_bullet_asteroid_collisions(self) -> None:
        for asteroid in self._bullets_vs_asteroids(self.player.bullet_manager):
            self.score_manager.add(Asteroid.points(asteroid.size))

            self.asteroids_destroyed += 1
            self.accuracy_hits += 1

    def check_saucer_bullet_asteroid_collisions(self, saucer: Saucer) -> None:
        self._bullets_vs_asteroids(saucer.bullet_manager)

    def check_bullet_saucer_collision(self, saucer: Saucer) -> None:
        if not saucer.is_alive:
            return

        for bullet in list(self.player.bullet_manager.bullets):
            if self._touching(
                saucer.x, saucer.y, saucer.radius, bullet.x, bullet.y, bullet.RADIUS
            ):
                self.player.bullet_manager.remove(bullet)

                saucer.die()

                self.score_manager.add(Saucer.points(saucer.size_type))

                self.saucers_destroyed += 1
                self.accuracy_hits += 1

                return

    def check_saucer_bullets_player_collision(self, saucer: Saucer) -> None:
        if not self.player.is_active:
            return

        for bullet in list(saucer.bullet_manager.bullets):
            if self._touching(
                self.player.x, self.player.y, self.player.RADIUS,
                bullet.x, bullet.y, bullet.RADIUS,
            ):
                saucer.bullet_manager.remove(bullet)

                self.player.die()

                return

    def check_player_asteroid_collisions(self) -> None:
        if not self.player.is_active:
            return

        for asteroid in self.asteroid_manager.asteroids:
            if self._touching(
                asteroid.x, asteroid.y, asteroid.radius,
                self.player.x, self.player.y, self.player.RADIUS,
            ):
                self.score_manager.add(Asteroid.points(asteroid.size))
                self.asteroids_destroyed += 1

                self.asteroid_manager.remove(asteroid)

                self.player.die()

                return

    def check_saucer_player_collision(self, saucer: Saucer) -> None:
        if not self.player.is_active or not saucer.is_alive:
            return

        if self._touching(
            saucer.x, saucer.y, saucer.radius,
            self.player.x, self.player.y, self.player.RADIUS,
        ):
            self.score_manager.add(Saucer.points(saucer.size_type))
            self.saucers_destroyed += 1

            saucer.die()

            self.player.die()

    def check_saucer_asteroid_collision(self, saucer: Saucer) -> None:
        if not saucer.is_alive:
            return

        for asteroid in self.asteroid_manager.asteroids:
            if self._touching(
                asteroid.x, asteroid.y, asteroid.radius,
                saucer.x, saucer.y, saucer.radius,
            ):
                self.asteroid_manager.remove(asteroid)

                saucer.die()

                return

    def update(self, saucer: Saucer | None) -> None:
        self.check_bullet_asteroid_collisions()

        if saucer is not None:
            self.check_bullet_saucer_collision(saucer)
            self.check_saucer_bullet_asteroid_collisions(saucer)
            self.check_saucer_bullets_player_collision(saucer)

        self.check_player_asteroid_collisions()

        if saucer is not None:
            self.check_saucer_player_collision(saucer)
            self.check_saucer_asteroid_collision(saucer)
