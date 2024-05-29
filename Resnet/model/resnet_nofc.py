# Credit to https://github.com/bupt-ai-cz/HHCL-ReID/blob/main/hhcl/models/resnet.py
from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']



class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        print('pooling_type: {}'.format(pooling_type))
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)

        # 减小大深度的resnet的第四层的下采样率，使提取到的特征维度更大
        if self.depth >= 50:
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)

        # 按Resnet原文，但是不包括最后的全连接层
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        # 这篇文章开发了多种池化层，默认使用平均池化
        self.gap = build_pooling_layer(pooling_type)

        # 如果不在全连接层之后截断，添加新的全连接层
        if not self.cut_at_pooling:
            self.num_features = num_features       # 自定义MLP的输入维度，如果为0则使用原版resnet输入维度
            self.norm = norm                       # 归一化
            self.dropout = dropout                 # dropout
            self.has_embedding = num_features > 0  #
            self.num_classes = num_classes         # 类别数

            # 原版resnet的全连接层的输入维度
            out_planes = resnet.fc.in_features     # 2048

            # Append new layers
            # 如果使用embedding，添加一个全连接层和一个BatchNorm层
            # 否则直接和分类器相连
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

            # 添加一个全连接层, 用于分类
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    """
    backbone -> pooling -> batch_norm -> classifier
    """
    def forward(self, x):
        # bs = x.size(0)
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        # if used the is training
        # if (self.training is False):
        #     bn_x = F.normalize(bn_x)
            # return bn_x

        # norm false
        if self.norm:
            bn_x = F.normalize(bn_x)
        # has_embedding false
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        # npdropout
        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob, bn_x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)