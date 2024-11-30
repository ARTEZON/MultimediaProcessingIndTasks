import pygame
import ctypes
from window import Window
from webcam import Webcam
import scenes


START_FULLSCREEN = False
WIDTH, HEIGHT = 1200, 900
MAX_FPS = 60

MIRROR_CAMERA = True
GOD_MODE = True
TRACKING_MODE = 'mosse'  # canny, diff, mosse

CANNY_THRESHOLD_LOW = 30
CANNY_THRESHOLD_HIGH = 60
CANNY_PREPROCESS_BLUR_KERNEL_SIZE = 11
CANNY_DILATE_EDGES = 5  # 25
CANNY_ERODE_EDGES = 0  # 10

DIFF_THRESHOLD = 32


class Game:
    def __init__(self):
        pygame.init()
        self.window = Window(WIDTH, HEIGHT, START_FULLSCREEN, 'Popping Bubbles!')
        self.display = self.window.display
        self.tracking_mode = TRACKING_MODE
        self.webcam = None
        self.webcam_connect(0)
        self.scene = scenes.MainMenu(self, self.webcam)
        self.can_lose = not GOD_MODE

        self.clock = pygame.time.Clock()
        self.fps = 0
        self.run = True
        while self.run:
            self.process_events()
            self.scene.update()
            pygame.display.flip()
            self.dt = self.clock.tick(MAX_FPS)
            self.fps = self.clock.get_fps()
        pygame.quit()

    def webcam_connect(self, camera_id):
        self.webcam = Webcam(self, camera_id, MIRROR_CAMERA, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH, CANNY_PREPROCESS_BLUR_KERNEL_SIZE, CANNY_DILATE_EDGES, CANNY_ERODE_EDGES, DIFF_THRESHOLD)

    def webcam_disconnect(self):
        self.webcam = None

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
            elif event.type == pygame.KEYDOWN:
                self.process_keys(event.key)
            elif event.type == pygame.VIDEORESIZE:
                if not self.window.fullscreen:
                    self.window._w = event.w
                    self.window._h = event.h
            self.scene.process_event(event)

    def process_keys(self, key):
        if key == pygame.K_f or key == pygame.K_F11:
            self.window.toggle_fullscreen()
        self.scene.process_key(key)

    def error_message(self, msg):
        print(msg)


if __name__ == '__main__':
    ctypes.windll.user32.SetProcessDPIAware()
    game = Game()
