import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """
    Advanced Inception-like Module with six parallel branches for rich, multiscale feature extraction.
    Includes standard convolutions, large-kernel convolutions, and dilated convolutions.
    """

    def __init__(
            self,
            in_channels: int,
            out_1x1: int,
            out_3x3_reduce: int,
            out_3x3: int,
            out_5x5_reduce: int,
            out_5x5: int,
            out_7x7_reduce: int,
            out_7x7: int,
            dilated_3x3_reduce: int,
            dilated_out_3x3: int,
            pool_projection: int) -> None:
        super(InceptionModule, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.GELU(approximate='tanh')
        )

        # Branch 2: 3x3 convolution
        self.branch2: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_3x3_reduce),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(out_3x3_reduce, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.GELU(approximate='tanh')
        )

        # Branch 3: 5x5 convolution
        self.branch3: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out_5x5_reduce),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(out_5x5_reduce, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.GELU(approximate='tanh')
        )

        # Branch 4: 7x7 convolution
        self.branch4: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, out_7x7_reduce, kernel_size=1),
            nn.BatchNorm2d(out_7x7_reduce),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(out_7x7_reduce, out_7x7, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_7x7),
            nn.GELU(approximate='tanh')
        )

        # Branch 5: Dilated 3x3 convolution for wider receptive field
        self.branch5: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, dilated_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(dilated_3x3_reduce),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(dilated_3x3_reduce, dilated_out_3x3, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(dilated_out_3x3),
            nn.GELU(approximate='tanh')
        )

        # Branch 6: Max pooling
        self.branch6: nn.Module = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_projection, kernel_size=1),
            nn.BatchNorm2d(pool_projection),
            nn.GELU(approximate='tanh')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
                self.branch5(x),
                self.branch6(x)
            ],
            dim=1
        )
