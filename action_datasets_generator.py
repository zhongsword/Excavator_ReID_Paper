import os
import time

import cv2
import numpy
import numpy as np

from utils.io import BaseTrackResultReader


def bbox_scare(box, im_width, im_height, scale=1.1):
    x, y, w, h = box
    x = x + w / 2
    y = y + h / 2
    w = w * scale
    h = h * scale
    x1 = max(0, x - w / 2)
    x2 = min(im_width, x + w / 2)
    y1 = max(0, y - h / 2)
    y2 = min(im_height, y + h / 2)
    return np.array([x1, y1, x2, y2], dtype=int)


def img_resize(img, size):
    h, w = img.shape[:2]

    if h < w:
        resized_img = cv2.resize(img, (size, int(size * h / w)), interpolation=cv2.INTER_AREA)
        padded_height = size - resized_img.shape[0]
        pad_top = int(padded_height / 2)
        pad_bottom = padded_height - pad_top
        pad_color = np.full((padded_height, size, 3), (0, 0, 0), dtype=np.uint8)
        img_padded = np.concatenate((pad_color[:pad_top, :, :], resized_img, pad_color[pad_top:, :, :]), axis=0)
    else:
        resized_img = cv2.resize(img, (int(size * w / h), size), interpolation=cv2.INTER_AREA)
        padded_width = size - resized_img.shape[1]
        pad_left = int(padded_width / 2)
        pad_right = padded_width - pad_left
        pad_color = np.full((size, padded_width, 3), (0, 0, 0), dtype=np.uint8)
        img_padded = np.concatenate((pad_color[:, :pad_left, :], resized_img, pad_color[:, pad_left:, :]), axis=1)
    return img_padded


class Dir_mergr:

    def __init__(self, name, length):
        os.makedirs(name=name, exist_ok=True)
        self.save_dir = name
        self.length_range = length
        self.cont = 0

    def __call__(self, img):
        cv2.imwrite(os.path.join(self.save_dir, f'{time.time()}.png'), img)
        self.cont += 1
        if self.cont == self.length_range:
            return "full"
        else:
            return False


class Video_merge:

    def __init__(self, name, size):
        os.makedirs(name=name, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        self.video = cv2.VideoWriter(os.path.join(name, f'{time.time()}.mp4'), fourcc, 15, (size, size))
        self.size = size

    def __call__(self, img, end_trigger=False):
        if end_trigger:
            self.video.release()
            return
        img = img_resize(img, self.size)
        self.video.write(img)


class Action_datasets_generator(BaseTrackResultReader):
    videos = {}

    def __init__(self, video_path, track_results, video_length):
        super().__init__(video_path, track_results)
        self.task_path = os.path.join(os.getcwd(), f'{time.time()}')
        self.video_length = video_length

    def _work(self, img: numpy.array, results: list):
        results = np.array(results)
        boxes = np.array(results[:, 2:6], dtype=float)
        boxes = np.apply_along_axis(bbox_scare, axis=1, arr=boxes,
                                    im_width=self.im_width, im_height=self.im_height)
        for track_id, box in zip(results[:, 1], boxes):
            if track_id not in self.videos:

                self.videos[track_id] = Video_merge(name=os.path.join(self.task_path, track_id),
                                                    size=640)

            self.videos[track_id](img[box[1]:box[3], box[0]:box[2]])

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        for _, v in self.videos.items():
            v(img=None, end_trigger=True)


if __name__ == "__main__":
    with Action_datasets_generator("121241-123840.mp4", "results.txt", 16) as ac:
        ac.rus()
