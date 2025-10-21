import torch
import torch.nn as nn

from squash import squash


class PrimaryCapsule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dim_caps: int,
            kernel_size: int = 9,
            stride: int = 2) -> None:
        super().__init__()
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0
                )
                for _ in range(dim_caps)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [caps(x).view(x.size(0), -1, 1) for caps in self.capsules]
        outputs = torch.cat(outputs, dim=-1)
        return squash(outputs)
