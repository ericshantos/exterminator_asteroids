import pygame

from configs import cfg

from .hud import HUD


class Renderer:
    def __init__(self, world: "GameWorld") -> None:
        pygame.init()

        self.screen: pygame.Surface = pygame.display.set_mode(
            (world.space.width, world.space.height)
        )

        self.clock: pygame.time.Clock = pygame.time.Clock()

        pygame.display.set_caption(cfg.screen.window_title)

        self.hud = HUD(self.screen)

        self.world = world

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        return True

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))

        self.world.player.draw(self.screen)

        self.world.asteroid_manager.draw(self.screen)

        self.world.saucer_manager.draw(self.screen)

        self.hud.draw_lives(self.world.player)
        self.hud.draw_score(self.world.score_manager.get_score())

        pygame.display.flip()

        self.clock.tick(cfg.screen.fps)
