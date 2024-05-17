import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import sys

from YOLOv8.predictor import Detector

from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

from utils.parser import get_config



class VideoTracker(object):
    
    def __init__(self, cfg, video_path, **args) :
        """sumary_line
        
        Keyword arguments:
        cfg:(str) path to config for initialing tracker
        video_path:(str) used video.
        args:
            yolo_weights:(str) path to the yolo weights.
            save_path:(str) path for saving the result.
            frame_interval:(int) frame interval for tracking.
            display:(bool) show the result if True.
            use_cuda:(bool) if use the GPU or not.
            
        Return: return_description
        """
        
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args["use_cuda"] and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args["display"]:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args["display_width"], args["display_height"])


        self.vdo = cv2.VideoCapture()
        self.detector = Detector(self.args["yolo_weights"])
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = "excavator"
  
    
    def __enter__(self):
        assert os.path.isfile(self.video_path), "Path error"
        self.vdo.open(self.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vdo.isOpened()
        
        if self.args["save_path"]:
            os.makedirs(self.args["save_path"], exist_ok=True)
            
            self.save_video_path = os.path.join(self.args["save_path"], "results.avi")
            self.save_results_path = os.path.join(self.args["save_path"], "results.txt")
            self.save_images_path = os.path.join(self.args["save_path"], "images")
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))
            
            self.logger.info("Save results to {}".format(self.args["save_path"]))
            
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)        
            
    
    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args["frame_interval"]:
                continue
            
            start = time.time()            
            _, ori_im = self.vdo.retrieve()
            # im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = ori_im
            
            #dectection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            
            mask=cls_ids == 0
            bbox_xywh = bbox_xywh[mask]
            # # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            # bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            
            outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)
            
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()         
            
            if self.args["display"]:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args["save_path"]:
                self.writer.write(ori_im)

            if self.args["save_path"]:
            # save results
                write_results(self.save_results_path, results, 'mot')

            if self.args["save_frame"]:
                os.makedirs(self.save_images_path, exist_ok=True)
                cv2.imwrite(os.path.join(self.save_images_path, f'{idx_frame}.jpg'), ori_im)

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))           
            
            
if __name__ == "__main__":
    cfg = get_config()
    cfg.merge_from_file("/home/zlj/Excavator_ReID/configs/deep_sort.yaml")
    
    with VideoTracker(cfg, 
                      video_path="/mnt/zlj-own-disk/No93Video/123905-130643.mp4",
                      yolo_weights="/home/zlj/ultralytics/runs/detect/train29/weights/best.pt",
                      frame_interval=3,
                      display=False,
                      display_width=720,
                      display_height=360,
                      use_cuda=True,
                      save_path='./2_ori_fi3',
                      save_frame=False) as vdo_trk:
        vdo_trk.run()



"""
Iou和外观特征比例的修改放在了nn_matching.py中
gate的修改在tracker.py的_match函数中
"""