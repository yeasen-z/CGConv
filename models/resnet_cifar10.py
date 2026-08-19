"""CIFAR ResNet variants used by the CGConv experiments."""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.nn import init

from cgToolkit import CGConv2d

__all__ = [
    "CifarResNet",
    "ResNetBasicblock",
    "cgc_resnet32",
    "cgc_resnet56",
    "cgc_resnet110",
    "resnet20",
    "resnet44",
    "resnet56",
]


class DownsampleA(nn.Module):
    """CIFAR ResNet option-A downsampling."""

    def __init__(self, n_in: int, n_out: int, stride: int):
        super().__init__()
        if stride != 2 or n_out != 2 * n_in:
            raise ValueError("DownsampleA expects stride=2 and n_out=2*n_in")
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, torch.zeros_like(x)), dim=1)


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cgc: bool = False,
    ):
        super().__init__()
        conv = CGConv2d if cgc else nn.Conv2d
        self.conv_a = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn_a = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_b = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_b = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu1(self.bn_a(self.conv_a(x)))
        out = self.bn_b(self.conv_b(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu2(residual + out)


class CifarResNet(nn.Module):
    """ResNet for 32x32 images, with optional CGConv blocks.

    ``norm_count`` gives the number of ordinary convolution blocks at the
    beginning of each stage. Remaining blocks use ``CGConv2d``.
    """

    def __init__(
        self,
        block,
        depth: int,
        num_classes: int,
        norm_count: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must be one of 20, 32, 44, 56, 110, ...")
        blocks_per_stage = (depth - 2) // 6

        if norm_count is None:
            norm_count = (blocks_per_stage,) * 3
        if len(norm_count) != 3 or any(
            count < 0 or count > blocks_per_stage for count in norm_count
        ):
            raise ValueError(
                "norm_count must contain three values between 0 and "
                f"{blocks_per_stage}"
            )

        self.depth = depth
        self.num_classes = num_classes
        self.norm_count = tuple(norm_count)
        self.inplanes = 16

        self.conv_1_3x3 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu_1 = nn.ReLU(inplace=True)
        self.stage_1 = self._make_layer(
            block, 16, blocks_per_stage, stride=1, norm_num=norm_count[0]
        )
        self.stage_2 = self._make_layer(
            block, 32, blocks_per_stage, stride=2, norm_num=norm_count[1]
        )
        self.stage_3 = self._make_layer(
            block, 64, blocks_per_stage, stride=2, norm_num=norm_count[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride, norm_num):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(
                self.inplanes, planes * block.expansion, stride
            )

        layers = []
        for block_index in range(blocks):
            block_stride = stride if block_index == 0 else 1
            block_downsample = downsample if block_index == 0 else None
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    block_stride,
                    block_downsample,
                    cgc=block_index >= norm_num,
                )
            )
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.out_channels
                )
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.relu_1(self.bn_1(self.conv_1_3x3(x)))
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        return self.classifier(torch.flatten(x, 1))


def resnet20(num_classes: int = 10):
    return CifarResNet(ResNetBasicblock, 20, num_classes)


def resnet44(num_classes: int = 10):
    return CifarResNet(ResNetBasicblock, 44, num_classes)


def resnet56(num_classes: int = 10):
    return CifarResNet(ResNetBasicblock, 56, num_classes)


def cgc_resnet32(num_classes: int = 10, norm_count=(2, 2, 2)):
    return CifarResNet(
        ResNetBasicblock, 32, num_classes, norm_count=norm_count
    )


def cgc_resnet56(num_classes: int = 10, norm_count=(3, 3, 3)):
    return CifarResNet(
        ResNetBasicblock, 56, num_classes, norm_count=norm_count
    )


def cgc_resnet110(num_classes: int = 10, norm_count=(3, 3, 3)):
    return CifarResNet(
        ResNetBasicblock, 110, num_classes, norm_count=norm_count
    )
