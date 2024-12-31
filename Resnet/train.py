import os.path

import numpy as np

from utils.data_loader import DataLoader
from model import resnet_nofc
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from collections import OrderedDict

# 设备
devices = [0, 1]

# 数据
datasets = DataLoader('/mnt/zlj-own-disk/imgs', 512, 40)

# 存储地址
output_reid = '/home/zlj/Excavator_ReID/Resnet/Checkpoints/'

# 模型
model = resnet_nofc.resnet50(num_classes=len(datasets.class_names))
model.classifier.requires_grad_(False)  # 冻结MLP的参数，只训练resnet的参数
model = nn.DataParallel(model, device_ids=devices)
# device = torch.device('cuda:0')
model = model.to(devices[0])

# 损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#训练
save_dir = os.path.join(output_reid, f"{time.time()}")
os.makedirs(save_dir, exist_ok=True)
epochs = 40
LOSS = 20

# # resume_train
# save_dir = os.path.join(output_reid, '1717512518.453511')
# checkpoint_path = os.path.join(save_dir, 'last.pth')
# epochs = 100
# checkpoint = torch.load(checkpoint_path)
# # new_state_dict = OrderedDict()
# # for k, v in checkpoint.items():
# #     name = k[7:]  # 去除 'module.' 前缀
# #     new_state_dict[name] = v
# model.load_state_dict(checkpoint)
# LOSS = 1.1719

train_log = np.zeros(shape=(epochs, 2))
val_log = np.zeros(shape=(epochs, 2))


for epoch in range(epochs):

    for phase in ('train', 'val'):
        if phase == 'train':
            model.train()
        else:
            model.eval()

        # 测试
        # if phase == 'train':
        #     print('Testing, train escape...')
        #     continue

        running_loss = 0.0
        running_corrects = 0

        # 迭代数据
        for inputs, labels in tqdm(datasets.dataloaders[phase],
                                   desc="epoch %d/%d" % (epoch, epochs)):
            inputs = inputs.to(devices[0])
            labels = labels.to(devices[0])

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs[0], 1)
                loss = criterion(outputs[0], labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(datasets.image_datasets[phase])
        epoch_acc = running_corrects.double() / len(datasets.image_datasets[phase])

        tqdm.write(f'{phase.capitalize()} LOSS: {epoch_loss:.4f} ACC: {epoch_acc:.4f}')
        if phase == 'train':
            train_log[epoch, 0] = epoch_loss
            train_log[epoch, 1] = epoch_acc
        else:
            val_log[epoch, 0] = epoch_loss
            val_log[epoch, 1] = epoch_acc
        if phase == "val":
            torch.save(model.state_dict(), os.path.join(save_dir, 'last.pth'))
            if epoch_loss < LOSS:
                LOSS = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
                np.save(os.path.join(save_dir, 'train_log.npy'), train_log)
                np.save(os.path.join(save_dir, 'val_log.npy'), val_log)
