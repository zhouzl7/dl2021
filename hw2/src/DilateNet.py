# encoding: utf-8
"""
@author: zhou zelong
@contact: zzl850783164@163.com
@time: 2021/4/13 14:43
@file: DilateNet.py
@desc: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    dilation = 1

    def __init__(self, in_planes, planes, stride=(1, 1)):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride,
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Increase the receptive field without increasing the amount of calculation
class DilateBlock(nn.Module):
    dilation = 1

    def __init__(self, in_planes, planes, stride=(1, 1)):
        super(DilateBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(in_planes, planes // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv1_2 = nn.Conv2d(in_planes, planes // 4, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False, dilation=2)
        self.conv1_3 = nn.Conv2d(in_planes, planes // 4, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False, dilation=3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=stride,
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.cat([self.conv1_1(x), self.conv1_2(x), self.conv1_3(x)], dim=1)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DilateNet(nn.Module):
    def __init__(self, block1, block2, num_blocks, num_classes=10):
        super(DilateNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.layer1 = self._make_layer(block1, 64, num_blocks[0], stride=(1, 1))
        self.layer2 = self._make_layer(block1, 128, num_blocks[1], stride=(2, 2))
        self.layer3 = self._make_layer(block1, 256, num_blocks[2], stride=(2, 2))
        self.layer4 = self._make_layer(block1, 512, num_blocks[3], stride=(2, 2))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.dilation

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])

        out_t = out.view(out.size(0), -1)
        out = self.linear(out_t)
        return out


def dilateNet18(num_classes):
    return DilateNet(BasicBlock, DilateBlock, [2, 2, 2, 2], num_classes)


def dilateNet34(num_classes):
    return DilateNet(BasicBlock, DilateBlock, [3, 4, 6, 3], num_classes)
