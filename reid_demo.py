import cv2
import numba
import numpy as np
import os
from YOLOv8.predictor import Segmentor
from utils.io import BaseTrackResultReader
from Resnet.predictor import BaseExtractor
from Resnet.feature_save import Feature_map
from Resnet.feature_save import Feature_bank
import numpy
import json
import time

import faulthandler
faulthandler.enable()

def list_duplicates(ls_in, num=1):
    """
    横向复制扩展列表中的元素
    """
    res = []
    for i in ls_in:
        for _ in range(num):
            res.append(i)
    return res

class reid_feature_extractor(BaseTrackResultReader):

    def __init__(self, video_path, track_result_path, seg_weight_path, batch=None, save_path=None):
        super().__init__(video_path, track_result_path, batch=batch)
        self.segmentor = Segmentor(seg_weight_path, device='cuda:0')
        self.extractor = BaseExtractor(device='cuda:1')
        self.feature_map = Feature_map()
        self.feature_bank = Feature_bank()
        self.worker_done = False
        if save_path is None:
            save_path = "feature_result"
            save_dir = os.path.join(os.getcwd(), save_path)
            os.makedirs(save_dir, exist_ok=True)
            time_index = time.time()
            self.map_sava_path = os.path.join(save_dir, "{time}_map.npy".format(time=time_index))
            self.bank_sava_path = os.path.join(save_dir, "{time}_bank.npy".format(time=time_index))

    def bbox_scare(self, box, scale=1.1):
        x, y, w, h = box
        x = x + w / 2
        y = y + h / 2
        w = w * scale
        h = h * scale
        x1 = max(0, x - w / 2)
        x2 = min(self.im_width, x + w / 2)
        y1 = max(0, y - h / 2)
        y2 = min(self.im_height, y + h / 2)
        return numpy.array([x1, y1, x2, y2])

    def _tail_work(self):
        if not self.frame_ids:
            return
        res = self.segmentor(self.frame_ids, self.imgs, self.boxes, self.track_ids)
        self._batch_reset()
        self.worker_done = True

    def _batch_work(self, img, results):
        results = numpy.array(results)
        self.imgs.append(img)
        self.frame_ids.append(results[:, 0])
        self.track_ids.append(results[:, 1])
        boxes = numpy.array(results[:, 2:6], dtype=float)
        boxes = np.apply_along_axis(self.bbox_scare, 1, boxes)
        self.boxes.append(boxes)
        if len(self.frame_ids) < self.batch:
            return
        else:
            res = self.segmentor(self.frame_ids, self.imgs, self.boxes, self.track_ids)
            if res:
                features = self.extractor(res[2])
                self.feature_map(res[0], res[1], features)
                self.feature_bank(res[0], res[1], features)
            self._batch_reset()

    def _work(self, img, results):
        frame_id = results[0][0]
        track_ids = [x[1] for x in results]
        boxes = [[float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in results]
        boxes = [self.bbox_scare(*x) for x in boxes]
        res = self.segmentor(img, boxes, track_ids)
        for k, v in res.items():
            features = []
            for k1, v1 in v.items():
                if v1.size == 0:
                    features.append(numpy.zeros(2048))
                else:
                    feature = self.extractor(v1)
                    features.append(feature)
            self.feature_bank(k, frame_id, features)
            self.feature_map(k, frame_id, features)
        self.worker_done = True

    @property
    def bank(self):
        if self.worker_done:
            return self.feature_bank.result
        else:
            print("worker not done!")
            return None

    @property
    def map(self):
        if self.worker_done:
            return self.feature_map.result
        else:
            print("worker not done!")
            return None


if __name__ == "__main__":
    with reid_feature_extractor(video_path='/home/zlj/Excavator_ReID/121241-123840.mp4',
                                seg_weight_path='/home/zlj/ultralytics/runs/segment/train3/weights/best.pt',
                                track_result_path='/home/zlj/Excavator_ReID/results.txt',
                                batch=4) as extractor:
        extractor.rus()
        ...
        numpy.save(extractor.map_sava_path, extractor.map)
        numpy.save(extractor.bank_sava_path, extractor.bank)
        # json.dump(extractor.map[0], open('feature_map.json', 'w'))
        # json.dump(extractor.map[1], open('map_time.json', 'w'))
        # json.dump(extractor.bank[0], open('feature_bank.json', 'w'))
        # json.dump(extractor.bank[1], open('bank_time.json', 'w'))
