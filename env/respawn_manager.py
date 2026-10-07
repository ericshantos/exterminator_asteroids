from entities import Asteroid, Saucer

from .toroidal_space import ToroidalSpace


class RespawnManager:
    SAFE_RADIUS: float = 120.0

    @classmethod
    def can_respawn(
        cls,
        space: ToroidalSpace,
        x: float,
        y: float,
        asteroids: list[Asteroid],
        saucer: Saucer | None = None,
    ) -> bool:
        for asteroid in asteroids:
            distance = space.distance(x, y, asteroid.x, asteroid.y)

            if distance <= cls.SAFE_RADIUS + asteroid.radius:
                return False

        if saucer is not None and saucer.is_alive:
            distance = space.distance(x, y, saucer.x, saucer.y)

            if distance <= cls.SAFE_RADIUS + saucer.radius:
                return False

        return True
