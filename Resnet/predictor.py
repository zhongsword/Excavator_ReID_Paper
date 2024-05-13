import cv2
import numpy
import numpy as np
from .model.resnet_nofc import resnet50
import torch
from torchvision import transforms
from PIL import Image


class BaseExtractor():
    def __init__(self, weight_path=None, device="cuda:0"):
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("cuda is not available, use cpu instead")
        else:
            self.device = torch.device(device)
        self.model = resnet50(cut_at_pooling=True)
        if weight_path:
            state_dict = torch.load(weight_path)
            self.model.load_state_dict(state_dict)

    @property
    def _img_transform(self):
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img: numpy.ndarray):
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print()
        img = Image.fromarray(img)
        img_tensor = self._img_transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        self.model.to(self.device)
        ret = self.model(img_tensor).cpu().detach().numpy().flatten()
        return ret

# if __name__ == "__main__":
#     img = Image.open(f'/mnt/zlj-own-disk/segmentation_datasets/1/images/11102.jpg')
#     img2 = Image.open(f'/mnt/zlj-own-disk/segmentation_datasets/1/images/11105.jpg')
#     exc = BaseExtractor()
#     ret1 = exc(img)
#     ret2 = exc(img2)
#     cosine_distance = 1 - np.dot(ret1, ret2) / (np.linalg.norm(ret1) * np.linalg.norm(ret2))
#     print(cosine_distance)
