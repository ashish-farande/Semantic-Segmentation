import torch
import torch.nn as nn
import torchvision.transforms.functional
from torch import Tensor


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1, dilation=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1_1: int,
            ch3_3_bottleneck: int,
            ch3_3: int,
            ch5_5r_bottleneck: int,
            ch5_5: int,
            pool_proj: int
    ) -> None:
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, ch1_1, kernel_size=1, padding=0)

        self.branch2 = nn.Sequential(
            # BasicConv2d(in_channels, ch3_3_bottleneck, kernel_size=1),
            BasicConv2d(in_channels, ch3_3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            # BasicConv2d(in_channels, ch5_5r_bottleneck, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            BasicConv2d(in_channels, ch5_5, kernel_size=3, padding=1),
            BasicConv2d(ch5_5, ch5_5, kernel_size=3, padding=1),
        )


    def forward(self, x: Tensor):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs, 1)


class HAPNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Stem
        # 786 X 384 X 3
        self.c1 = BasicConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 786 X 384 X 16
        self.c2 = BasicConv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Contraction
        # 786 X 384 X 32
        self.i1 = Inception(32, 16, 32, 32, 16, 16, 32)
        # 786 X 384 X 64
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # 384 X 192 x 64
        self.i2 = Inception(64, 32, 64, 64, 32, 32, 64)
        # 384 X 192 x 128
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # 192 X 96 x 128
        self.i3 = Inception(128, 32, 64, 64, 32, 32, 64)
        # 192 X 96 x 128
        self.m3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # 96 X 48 x 128
        self.i4 = Inception(128, 32, 64, 64, 32, 32, 642)
        # 96 X 48 x 128
        self.m4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Expansion
        # 48 X 24 x 128
        self.u1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, dilation=1, output_padding=0)
        # 96 X 48 x 128
        self.ie1 = Inception(128,32, 64, 64, 32, 32, 64)
        # 96 X 48 x 128
        self.u2 = nn.ConvTranspose2d( 128, 128, kernel_size=3, stride=2, dilation=1, output_padding=0)
        # 192 X 96 x 128
        self.ie2 = Inception(128, 32, 64, 64, 32, 32, 64)
        # 192 X 96 x 128
        self.u3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, dilation=1, output_padding=0)
        # 384 X 192 x 128
        self.ie3 = Inception(128, 32, 64, 64, 32, 32, 64)
        # 384 X 192 x 64
        self.u4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, dilation=1, output_padding=0)
        # 786 X 384 X 64
        self.ie4 = Inception(64, 16, 32, 32, 16, 16, 32)
        # 786 X 384 X 32
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)


    def forward(self, x):
            # Stem
            x = self.c2(self.c1(x))

            # Encoder
            x = self.m1(self.i1(x))
            x = self.m2(self.i2(x))
            x = self.m3(self.i3(x))
            x = self.m4(self.i4(x))

            # Decoder
            x = self.ie1(torchvision.transforms.functional.crop(self.u1(x), 1, 1, 48, 96))
            x = self.ie2(torchvision.transforms.functional.crop(self.u2(x), 1, 1, 96, 192))
            x = self.ie3(torchvision.transforms.functional.crop(self.u3(x), 1, 1, 192, 384))
            x = self.ie4(torchvision.transforms.functional.crop(self.u4(x), 1, 1, 384, 768))

            # Classifier
            score = self.classifier(x)
            return score

