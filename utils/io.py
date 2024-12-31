import os
import time
from typing import Dict

import numpy
import numpy as np
import cv2
from .log import get_logger
from tqdm import tqdm


# from data_utils.log import get_logger


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)


# def write_results(filename, results_dict: Dict, data_type: str):
#     if not filename:
#         return
#     path = os.path.dirname(filename)
#     if not os.path.exists(path):
#         os.makedirs(path)

#     if data_type in ('mot', 'mcmot', 'lab'):
#         save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
#     elif data_type == 'kitti':
#         save_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
#     else:
#         raise ValueError(data_type)

#     with open(filename, 'w') as f:
#         for frame_id, frame_data in results_dict.items():
#             if data_type == 'kitti':
#                 frame_id -= 1
#             for tlwh, track_id in frame_data:
#                 if track_id < 0:
#                     continue
#                 x1, y1, w, h = tlwh
#                 x2, y2 = x1 + w, y1 + h
#                 line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
#                 f.write(line)
#     logger.info('Save results to {}'.format(filename))

def build_reid_dataset(track_result_path, video_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    def get_results(f, last_result):
        results = []
        results.append(last_result)
        now_result = f.readline().split(',')
        while now_result[0] == last_result[0]:
            if now_result[0] == '':
                break
            results.append(now_result)
            last_result = now_result
            now_result = f.readline().split(',')
        return results, now_result

    vdo = cv2.VideoCapture()

    def output_image(results, ori_im, save_path):
        # result: [frame_id, id, x1, y1, w, h]
        for result in results:
            x = float(result[2]) + float(result[4]) / 2
            y = float(result[3]) + float(result[5]) / 2
            # 稍微将bbox扩大
            w = float(result[4]) * 1.1
            h = float(result[5]) * 1.1
            if w < 100 or h < 100:
                return

            x1 = max(0, int(x - w / 2))
            x2 = min(ori_im.shape[1], int(x + w / 2))
            y1 = max(0, int(y - h / 2))
            y2 = min(ori_im.shape[0], int(y + h / 2))
            img = ori_im[y1:y2, x1:x2]

            os.makedirs(os.path.join(save_path, result[1]), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, result[1], f'{result[0]}.jpg'), img)
            print(f"save {os.path.join(save_path, result[1], f'{result[0]}.jpg')}")

    track_results_f = open(track_result_path, 'r')
    results, last_result = get_results(track_results_f, track_results_f.readline().split(','))
    idx_frame = 0
    vdo.open(video_path)
    while vdo.grab():
        # print(vdo.grab())
        _, ori_im = vdo.retrieve()
        if results[0][0] == '':
            break
        if idx_frame != int(results[0][0]):
            idx_frame += 1
        else:
            output_image(results, ori_im, save_path)
            results, last_result = get_results(track_results_f, last_result)
            idx_frame += 1
    track_results_f.close()


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores


class BaseTrackResultReader:
    """
    The BaseTrackResultReader class is designed to facilitate reading and processing tracked object results from a
    specified video file and its corresponding tracking result file. It supports both sequential and batch processing
    modes, allowing for flexibility in managing and analyzing tracking data.

    To complete the class implementation, the following methods must be implemented:
        worker: The mian processing method for each frame of the video.
        batch_work: The processing method for batch data (if batch mode is on).
    """
    def __init__(self, video_path, track_result_path, data_type='mot', batch=None):
        """
        Parameters:
        - video_path (str): Path to the video file.
        - track_result_path (str): Path to the text file containing tracking results.
        - data_type (str, optional): Type of tracking data format; default is 'mot'.
        - batch (bool, optional): Enables or disables batch processing mode; default is None.

        Attributes Initialized:
        - vdo: A VideoCapture object to read frames from the video.
        - save_path: Directory path for saving processed content (if provided).
        - logger: Logger instance for logging messages.
        - batch_content: Placeholder for batch processing content.
        - frame_ids, track_ids, boxes, imgs: Lists to store batch data (if batch mode is on).
        """
        self.vdo = cv2.VideoCapture()
        self.video_path = video_path
        self.track_results_f = open(track_result_path, 'r')
        self.save_path = None
        self.logger = get_logger('root')
        self.batch = batch
        self.batch_content = None
        if self.batch:
            self.frame_ids = []
            self.track_ids = []
            self.boxes = []
            self.imgs = []

    def __enter__(self):
        assert os.path.isfile(self.video_path), f'Error: {self.video_path} is not a valid file.'
        self.vdo.open(self.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vdo.isOpened(), f'Error: Cannot open {self.video_path}.'
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            ...
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(exc_val)

    def _batch_reset(self):
        self.frame_ids.clear()
        self.track_ids.clear()
        self.boxes.clear()
        self.imgs.clear()

    def _batch_work(self, img: numpy.array, results: list):
        return img

    def _tail_work(self):
        ...

    def _work(self, img: numpy.array, results: list):
        return img

    def worker(self, img, results):
        if self.batch:
            self._batch_work(img, results)
        else:
            self._work(img, results)

    def _get_results(self, f, last_result):
        """
        read the track results in mot format.
        results: list of all the results in the same frame.
        last_result: the last result in the frame.
        info:
            result format:{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1
        """
        results = []
        results.append(last_result)
        now_result = f.readline().split(',')
        while now_result[0] == last_result[0]:
            if now_result[0] == '':
                break
            results.append(now_result)
            last_result = now_result
            now_result = f.readline().split(',')
        return results, now_result

    def rus(self):
        total_frame = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
        idx_frame = 0
        results, last_result = self._get_results(self.track_results_f, self.track_results_f.readline().split(','))
        with tqdm(total=total_frame, desc='Processing', unit='frame') as pbar:
            while self.vdo.grab():
                idx_frame += 1
                pbar.update(1)
                if results[0][0] == '':
                    break
                if idx_frame - 1 != int(results[0][0]):
                # if idx_frame - 1 != int(results[0][0]) or idx_frame < 37000:
                    continue
                else:
                    _, ori_im = self.vdo.retrieve()
                    res = self.worker(ori_im, results)
                    results, last_result = self._get_results(self.track_results_f, last_result)
                if results[0][0] == '':
                    break
                if idx_frame - 1 != int(results[0][0]):
                    continue


                else:
                    _, ori_im = self.vdo.retrieve()
                    res = self.worker(ori_im, results)
                    results, last_result = self._get_results(self.track_results_f, last_result)

            # batch没满的善后工作
            if self.batch:
                   self._tail_work()



if __name__ == "__main__":
    # reader = BaseTrackResultReader('/mnt/zlj-own-disk/No93Video/121241-123840.mp4', '/mnt/zlj-own-disk/fineDetector_results/my_fi9/results.txt')

    with BaseTrackResultReader('/mnt/zlj-own-disk/No93Video/153002-153717.mp4',
                               '/mnt/zlj-own-disk/fineDetector_results/my_fi9/results.txt',
                               batch=64) as test_track_reader:
        test_track_reader.rus()
