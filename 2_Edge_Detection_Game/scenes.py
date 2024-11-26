from abc import ABC, abstractmethod
import cv2
import pygame
from game_objects import Ball


class Scene(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def process_event(self, event):
        pass

    @abstractmethod
    def process_key(self, key):
        pass


class MainMenu(Scene):
    def __init__(self, game, webcam, text_lines=None):
        self.game = game
        self.window = self.game.window
        self.webcam = webcam

        self.text_size = None
        self.font = None
        self.update_menu()
        self.text_lines = ['WELCOME', '', 'Press SPACE to start'] if text_lines is None else text_lines

    def update(self):
        self.webcam.next_frame()
        camera_image = self.webcam.get_image(background=True)
        camera_surface = pygame.image.frombuffer(camera_image.tobytes(), camera_image.shape[1::-1], 'BGR')
        camera_surface = pygame.transform.scale(camera_surface, self.window.get_size())
        self.window.display.blit(camera_surface, (0, 0))

        self.show_text()

    def process_event(self, event):
        if event.type == pygame.VIDEORESIZE:
            self.update_menu()

    def process_key(self, key):
        if key == pygame.K_SPACE:
            self.game.scene = Gameplay(self.game, self.webcam)
        elif key == pygame.K_ESCAPE:
            self.game.run = False

    def update_menu(self):
        self.text_size = self.window.get_size()[1] // 20
        self.font = pygame.font.Font(pygame.font.get_default_font(), self.text_size)

    def show_text(self):
        for i, line in enumerate(self.text_lines):
            line_width = self.font.size(line)[0]
            text = self.font.render(line, True, (255, 255, 255))
            self.window.display.blit(text, ((self.window.get_size()[0] - line_width) // 2, (self.window.get_size()[1] - len(self.text_lines) * (self.text_size + 5)) // 2 + i * (self.text_size + 5)))


class Gameplay(Scene):
    def __init__(self, game, webcam):
        self.game = game
        self.window = self.game.window
        self.webcam = webcam
        self.display_edges = False
        self.score = 0
        self.speed = 1
        self.objects = []

        self.camera_image = None

        self.spawn_interval_msec = 200
        self.last_spawn_time = pygame.time.get_ticks()

        self.text_size = None
        self.font = None
        self.font_small = None
        self.update_overlay()

    def update(self):
        self.window.display.fill((0, 0, 0))
        self.webcam.next_frame()
        self.camera_image = self.webcam.get_image(edges=self.display_edges)
        if self.display_edges:
            self.camera_image = cv2.cvtColor(self.camera_image, cv2.COLOR_GRAY2BGR)
        camera_surface = pygame.image.frombuffer(self.camera_image.tobytes(), self.camera_image.shape[1::-1], 'BGR')
        camera_surface = pygame.transform.scale(camera_surface, self.window.get_size())
        self.window.display.blit(camera_surface, (0, 0))

        current_ticks = pygame.time.get_ticks()
        if current_ticks - self.last_spawn_time > self.spawn_interval_msec:
            self.last_spawn_time = current_ticks
            self.objects.append(Ball(self.window.get_size()))
            if self.speed < 8:
                self.speed += 0.003
            if self.spawn_interval_msec > 25:
                self.spawn_interval_msec -= 0.4

        for i in reversed(range(len(self.objects))):
            obj = self.objects[i]
            if obj.y > 1:
                del self.objects[i]
                if self.game.can_lose:
                    self.game.scene = MainMenu(self.game, self.webcam, ['GAME OVER!', f' Score: {self.score}', '', 'Press SPACE to play again'])
                else:
                    continue
            obj.x += obj.x_movement * self.window.get_size()[0]
            if obj.x_pixels < obj.radius_pixels and obj.x_movement < 0 or obj.x_pixels > self.window.get_size()[0] - obj.radius_pixels and obj.x_movement > 0:
                obj.x_movement = -obj.x_movement
            obj.y += 0.005 * self.speed + obj.radius * 0.001
            obj.set_rect(self.window.get_size())
            self.window.display.blit(obj.image, obj.rect)

        edges = self.webcam.get_image(edges=True)
        edges = cv2.resize(edges, self.window.get_size())
        for i in reversed(range(len(self.objects))):
            obj = self.objects[i]
            if 0 <= obj.y_pixels < edges.shape[0] and 0 <= obj.x_pixels < edges.shape[1]:
                if edges[int(obj.y_pixels), int(obj.x_pixels)] > 0:
                    self.score += 10 + int(obj.radius * 10)
                    del self.objects[i]

        self.show_overlay()

    def process_event(self, event):
        if event.type == pygame.VIDEORESIZE:
            self.update_overlay()
            for obj in self.objects:
                obj.set_size(self.window.get_size())

    def process_key(self, key):
        if key == pygame.K_ESCAPE:
            self.game.scene = MainMenu(self.game, self.webcam)
        elif key == pygame.K_c:
            self.display_edges = not self.display_edges

    def update_overlay(self):
        self.text_size = self.window.get_size()[1] // 15
        self.font = pygame.font.Font(pygame.font.get_default_font(), self.text_size)
        self.font_small = pygame.font.Font(pygame.font.get_default_font(), self.text_size // 3)

    def show_overlay(self):
        score = f"Score: {self.score}"
        text = self.font.render(score, True, (255, 255, 255))
        self.window.display.blit(text, (20, 20))
        fps = f"FPS: {int(self.game.fps)}"
        text = self.font_small.render(fps, True, (255, 255, 255))
        self.window.display.blit(text, (20, 110))
