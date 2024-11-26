import random
import colorsys
import pygame


# x, y: [0 .. 100] (% of the screen)

class Ball(pygame.sprite.Sprite):
    def __init__(self, window_size):
        super().__init__()
        self.radius = random.random()
        self.x = random.random()
        self.y = -self.radius / window_size[1]
        self.color_hsv = random.randrange(180), random.randrange(220, 250), random.randrange(200, 255)

        self.x_movement = (random.random() - 0.5) / 500000

        self.rect = None
        self.x_pixels = None
        self.y_pixels = None
        self.radius_pixels = None
        self.image_data = pygame.image.load("assets/ball.png").convert_alpha()
        self.image = None
        self.set_size(window_size)

    def set_color(self, color_hsv):
        color_rgb = [int(i * 255) for i in colorsys.hsv_to_rgb(*[i / 255 for i in color_hsv])]
        tint_surf = pygame.surface.Surface(self.rect.size)
        tint_surf.fill(color_rgb)
        self.image.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    def set_size(self, window_size):
        self.radius_pixels = (1 + self.radius) * window_size[1] * 0.03
        self.set_rect(window_size)
        self.image = pygame.transform.smoothscale(self.image_data, self.rect.size)
        self.set_color(self.color_hsv)

    def set_rect(self, window_size):
        self.x_pixels = self.x * window_size[0]
        self.y_pixels = self.y * window_size[1]
        self.rect = pygame.Rect(self.x_pixels - self.radius_pixels, self.y_pixels - self.radius_pixels,
                                self.radius_pixels * 2, self.radius_pixels * 2)
