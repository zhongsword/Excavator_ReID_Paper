import cv2
import numpy
import numpy as np
from .model.resnet_nofc import resnet50
import torch
from torchvision import transforms
from PIL import Image
from collections import OrderedDict

class BaseExtractor():
    def __init__(self, weight_path=None, device="cpu"):
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("cuda is not available, use cpu instead")
        else:
            self.device = torch.device(device)
        self.model = resnet50()
        if weight_path:
            state_dict = torch.load(weight_path)
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:]  # 去除 'module.' 前缀
            #     new_state_dict[name] = v
            state_dict.pop('classifier.weight')
            self.model.load_state_dict(state_dict)
            self.model.cut_at_pooling=True

    def list_divide(self, imglist):
        res_list = []
        for i in imglist:
            assert len(i) == 3
            for j in i:
                cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
                j = Image.fromarray(j)
                res_list.append(self._img_transform(j))
        return res_list

    @property
    def _img_transform(self):
        return transforms.Compose([
            transforms.Resize((348, 348)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, imgs: list()):
        ims_tensor_list = self.list_divide(imgs)
        # img_tensor = self._img_transform(img)
        img_tensors = torch.stack(ims_tensor_list, dim=0)
        img_tensors = img_tensors.to(self.device)
        self.model.to(self.device)
        ret = self.model(img_tensors).cpu().detach()
        ret = ret.view(int(ret.shape[0]/3), -1).numpy()
        assert len(ret)==len(imgs)
        return ret

# if __name__ == "__main__":
#     img = Image.open(f'/mnt/zlj-own-disk/segmentation_datasets/1/images/11102.jpg')
#     img2 = Image.open(f'/mnt/zlj-own-disk/segmentation_datasets/1/images/11105.jpg')
#     exc = BaseExtractor()
#     ret1 = exc(img)
#     ret2 = exc(img2)
#     cosine_distance = 1 - np.dot(ret1, ret2) / (np.linalg.norm(ret1) * np.linalg.norm(ret2))
#     print(cosine_distance)
