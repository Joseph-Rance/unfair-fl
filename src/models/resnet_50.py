"""Implementation of ResNet50

SOURCE: https://github.com/slkdfjslkjfd/fl_fairness_attacks/blob/attack/src/models/resnet_50.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):

    class Bottleneck(nn.Module):

        def __init__(self, ch_in: int, ch: int, s: int = 1) -> None:
            super(ResNet50.Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(ch_in, ch, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, s, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)
            self.conv3 = nn.Conv2d(ch, 4*ch, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(4*ch)

            if s != 1 or ch_in != 4*ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(ch_in, 4*ch, 1, s, bias=False),
                    nn.BatchNorm2d(4*ch)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x: torch.float) -> torch.float:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    def __init__(self) -> None:
        super(ResNet50, self).__init__()
        self.ch_in = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)
        self.linear = nn.Linear(2048, 10)

    def _make_layer(self, ch: int, n: int, stride: int) -> nn.Module:
        layers = []
        for s in [stride] + [1]*(n-1):
            layers.append(self.Bottleneck(self.ch_in, ch, s))
            self.ch_in = ch*4
        return nn.Sequential(*layers)

    def forward(self, x: torch.float) -> torch.float:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
