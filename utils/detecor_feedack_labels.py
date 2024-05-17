import os

from YOLOv8.predictor import Detector

img_path = "/mnt/zlj-own-disk/No93Video/North_Bund2"

detector = Detector(weights="/home/zlj/ultralytics/runs/detect/train14/weights/best.pt",
                    save_path="/mnt/zlj-own-disk/No93Video/labels")

img_list = os.listdir("/mnt/zlj-own-disk/No93Video/North_Bund2")

# print(img_list)

for img in img_list:
    img_path = os.path.join("/mnt/zlj-own-disk/No93Video/North_Bund2", img)
    detector(img_path)
    # print(img_path)