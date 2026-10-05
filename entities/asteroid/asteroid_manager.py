import pygame

from env.toroidal_space import ToroidalSpace

from .asteroid import Asteroid


class AsteroidManager:
    MAX_ASTEROIDS: int = 26

    def __init__(self, space: ToroidalSpace) -> None:
        self.space = space

        self.asteroids: list[Asteroid] = []
        self.debris: list[Asteroid] = []

    def spawn_wave(
        self,
        quantity: int,
        player_x: float,
        player_y: float,
        min_distance: float = 150.0,
    ) -> None:
        self.asteroids.clear()

        while len(self.asteroids) < quantity:
            asteroid = Asteroid(space=self.space)

            distance = self.space.distance(
                player_x, player_y, asteroid.x, asteroid.y
            )

            if distance < min_distance + asteroid.radius:
                continue

            self.asteroids.append(asteroid)

    def update(self) -> None:
        for asteroid in self.asteroids:
            asteroid.update()

        for fragment in self.debris:
            fragment.update()

        self.debris = [d for d in self.debris if not d.explosion_finished]

    def draw(self, surface: pygame.Surface) -> None:
        for fragment in self.debris:
            fragment.draw(surface)

        for asteroid in self.asteroids:
            asteroid.draw(surface)

    def split(self, asteroid: Asteroid) -> list[Asteroid]:
        if asteroid.size == 1:
            return []

        available_slots = self.MAX_ASTEROIDS - len(self.asteroids)

        quantity = max(0, min(2, available_slots))

        return asteroid.create_children(quantity)

    def remove(self, asteroid: Asteroid) -> None:
        if asteroid not in self.asteroids:
            return

        self.asteroids.remove(asteroid)

        children = self.split(asteroid)

        self.asteroids.extend(children)

        asteroid.trigger_explosion()
        self.debris.append(asteroid)

    def clear(self) -> None:
        self.asteroids.clear()
        self.debris.clear()

    @property
    def count(self) -> int:
        return len(self.asteroids)

    @property
    def is_empty(self) -> bool:
        return self.count == 0
