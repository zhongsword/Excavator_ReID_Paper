import torch
import torch.nn as nn
from .resnet_nofc import ResNet


class CNN4Action(ResNet):

    def __init__(self):
        super().__init__(depth=34, pretrained=False, cut_at_pooling=True)

class Pipline4Action(nn.Module):

    def __init__(self, num_classes, frame_seq_length=16):
        super().__init__()

        self.frame_seq_length = frame_seq_length

        self.cnn = CNN4Action()

        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=16, batch_first=True)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length, channels, height, width)
        assert self.frame_seq_length == x.size(1)
        batch_size = x.size(0)
        seq_length = x.size(1)

        # CNN, batch和seq_length一起卷积
        x = x.view(batch_size * seq_length, x.size(2), x.size(3), x.size(4))
        cnn_out = self.cnn(x)

        # 特征处理成LSTM可以接收的张量 (seq_length, batch_size, features)
        lstm_input = cnn_out.view(seq_length, batch_size, -1)

        lstm_output, _ = self.lstm(lstm_input)

        # 最后一个LSTM网络cell输出的作为分类的依据
        last_time_step = lstm_output[-1]

        out = self.classifier(last_time_step)

        return out

if __name__ == '__main__':
    x = torch.rand(size=(2, 32, 3, 256, 256))

    model = Pipline4Action(num_classes=3, frame_seq_length=32)
    device = torch.device('cuda:0')
    x = x.to(device)
    model.to(device)
    model(x)




