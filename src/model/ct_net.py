import torch
import torch.nn as nn

from ..modules.class_capsule import ClassCapsule
from ..modules.primary_capsule import PrimaryCapsule


class CTNet(nn.Module):
    """Enhanced Capsule Network for CT Scans."""

    def __init__(
            self,
            num_classes: int = 4,
            img_size: int = 64) -> None:
        super(CTNet, self).__init__()
        self.conv1: nn.Module = nn.Conv2d(
            in_channels=3,
            out_channels=384,
            kernel_size=9,
            stride=1
        )

        self.relu: nn.Module = nn.ReLU(inplace=True)
        self.primary_capsule: PrimaryCapsule = PrimaryCapsule(
            in_channels=384,
            out_channels=48,
            dim_caps=12,
            kernel_size=9,
            stride=2
        )

        # Dynamically calculate num_routes based on input image size
        conv1_out: int = img_size - 9 + 1
        primary_capsule_out: int = int((conv1_out - 9) / 2) + 1
        num_routes: int = 48 * (primary_capsule_out ** 2)

        self.class_capsule: ClassCapsule = ClassCapsule(
            num_capsules=num_classes,
            num_routes=num_routes,
            in_dim=12,
            out_dim=24,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.primary_capsule(x)

        batch_size = x.size(0)
        x = x.view(batch_size, -1, 12)
        x = self.class_capsule(x)

        return x
