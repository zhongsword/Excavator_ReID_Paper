from torchvision import datasets, transforms
import os
import torch

# 在训练集上:扩充 归一化
# 在验证集上:归一化

class DataLoader:
    def __init__(self, data_dir, imgsize, batch_size):

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(imgsize * 0.875),
                # transforms.Resize(imgsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(imgsize),
                transforms.CenterCrop(imgsize * 0.875),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            }

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),\
                                self.data_transforms[x])
                               for x in ['train', 'val']}

        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,\
                                                          shuffle=True, num_workers=4)
                                for x in ['train', 'val']}

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}

        self.class_names = self.image_datasets['train'].classes

if __name__ == '__main__':
    test_data = DataLoader()
    # TODO:





