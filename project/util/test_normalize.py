import numpy as np
import os
import cv2


def load_frames(file_dir):
    frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    frame_count = len(frames)
    buffer = np.empty((frame_count, 256, 256, 3), np.dtype('float32'))
    for i, frame_name in enumerate(frames):
        frame = np.array(cv2.imread(frame_name)).astype(np.float64)
        buffer[i] = frame

    return buffer

def normalize(buffer):
    for i, frame in enumerate(buffer):
        frame /= np.array([[[255.0, 255.0, 255.0]]])
        buffer[i] = frame

    return buffer

file_dir = r'D:\pythonProject\VideoSwinTransformer\util\000_003'
buffer = load_frames(file_dir)
print(buffer)
print(buffer.shape)
print("*******normalize***********")
buffer = normalize(buffer)
print(buffer)