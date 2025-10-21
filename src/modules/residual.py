import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Custom Residua Block to enable deeper network training.
    """
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.GELU(approximate='tanh'),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channels),
        )
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return self.gelu(x + out)