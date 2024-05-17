import os

from utils.io import build_reid_dataset

build_reid_dataset('/mnt/zlj-own-disk/fineDetector_results/my_fi3/results.txt',
            '/mnt/zlj-own-disk/No93Video/121241-123840.mp4',
                    os.path.join(os.getcwd(), 'reid_dataset_full'))