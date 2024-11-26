import pygame

class Window:
    def __init__(self, w: int, h: int, is_fullscreen: bool, title: str):
        self.display = None
        self._w: int = w
        self._h: int = h
        self.fullscreen: bool = is_fullscreen
        pygame.display.set_caption(title)
        self.set_mode(is_fullscreen)

    def get_size(self):
        return self.display.get_size()

    def toggle_fullscreen(self):
        self.set_mode(not self.fullscreen)

    def set_mode(self, fullscreen: bool):
        self.fullscreen = fullscreen
        if fullscreen:
            self.display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.display = pygame.display.set_mode((self._w, self._h), pygame.RESIZABLE)
