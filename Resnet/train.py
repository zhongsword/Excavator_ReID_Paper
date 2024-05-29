import os.path

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
# model = nn.DataParallel(model, device_ids=devices)
device = torch.device('cuda:0')
model = model.to(device)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 训练
# save_dir = os.path.join(output_reid, f"{time.time()}")
# os.makedirs(save_dir, exist_ok=True)
# epochs = 100
# LOSS = 20

# resume_train
save_dir = os.path.join(output_reid, '1716180655.7538185')
checkpoint_path = os.path.join(save_dir, 'last.pth')
epochs = 93
checkpoint = torch.load(checkpoint_path, map_location=device)
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:]  # 去除 'module.' 前缀
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
LOSS = 1.1719


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
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs[0], 1)
                loss = criterion(outputs[0], labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)
        epoch_loss = running_loss / len(datasets.image_datasets[phase])
        epoch_acc = running_corrects.double() / len(datasets.image_datasets[phase])

        tqdm.write(f'{phase.capitalize()} LOSS: {epoch_loss:.4f} ACC: {epoch_acc:.4f}')
        if phase == "val":
            torch.save(model.state_dict(), os.path.join(save_dir, 'last.pth'))
            if epoch_loss < LOSS:
                LOSS = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))

