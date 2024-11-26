import cv2
import numpy as np
import pygame


class Webcam:
    def __init__(self, camera_id: int, threshold_low, threshold_high, blur_kernel_size: int, dilate_edges: int, diff_mode: bool, diff_thresh: float):
        self.camera_id = camera_id
        self.capture = cv2.VideoCapture(self.camera_id)
        self.w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        if not self.capture.read()[0]:
            raise Exception(f'No camera found with id {self.camera_id}')
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.blur_kernel_size = blur_kernel_size
        self.last_frame = None
        self.last_frame_timestamp = 0
        self.prev_frame = None
        self.dilate_edges = dilate_edges

        self.processing = self.diff if diff_mode else self.canny
        self.diff_thresh = diff_thresh

    def next_frame(self):
        elapsed = pygame.time.get_ticks() - self.last_frame_timestamp
        if elapsed > 1000 / self.fps:
            ok, frame = self.capture.read()
            if not ok:
                return self.last_frame
            self.prev_frame = self.last_frame
            self.last_frame = frame
            self.last_frame_timestamp = pygame.time.get_ticks()
        return self.last_frame

    def get_image(self, edges=False, background=False):
        frame = self.last_frame
        if edges:
            frame = self.processing(frame)
        if background:
            frame = cv2.GaussianBlur(frame, (55, 55), 0)
            frame //= 2
        return frame

    def canny(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_kernel_size > 1 and self.blur_kernel_size % 2 == 1:
            frame = cv2.GaussianBlur(frame, (self.blur_kernel_size, self.blur_kernel_size), 0)
        frame = cv2.Canny(frame, self.threshold_low, self.threshold_high)
        if self.dilate_edges > 0:
            frame = cv2.dilate(frame, np.ones((self.dilate_edges, self.dilate_edges), np.uint8))
        return frame

    def diff(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        if self.blur_kernel_size > 1 and self.blur_kernel_size % 2 == 1:
            frame = cv2.GaussianBlur(frame, (self.blur_kernel_size, self.blur_kernel_size), 0)
            prev_frame = cv2.GaussianBlur(prev_frame, (self.blur_kernel_size, self.blur_kernel_size), 0)
        frame_diff = cv2.absdiff(prev_frame, frame)
        ret, frame_threshold = cv2.threshold(frame_diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        return frame_threshold

    def __del__(self):
        self.capture.release()
