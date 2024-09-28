import cv2
import os

directory = 'videos/'

for file in os.listdir(directory):
    if os.path.isfile(directory + file):
        cap = cv2.VideoCapture(directory + file)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            continue
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = chr(fourcc & 0xff) + chr((fourcc >> 8) & 0xff) + chr((fourcc >> 16) & 0xff) + chr((fourcc >> 24) & 0xff)
        print(f'{file} | {fourcc} | {w}x{h} | {fps} fps | {duration} seconds')