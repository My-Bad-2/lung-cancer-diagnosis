import torch
import torch.nn as nn

from inception import InceptionModule
from residual import ResidualBlock
from cbam import CBAM

# class HistoNet(nn.Module):
#     """
#     Custom CNN for Histopathological Images.
#     """
#
#     def __init__(self, num_classes: int = 3) -> None:
#         super(HistoNet, self).__init__()
#         self.stem: nn.Module = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=3),
#             nn.BatchNorm2d(num_features=96),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#
#         # Layer 1
#         self.inception1: InceptionModule = InceptionModule(
#             in_channels=96,
#             out_1x1=64,
#             out_3x3_reduce=64,
#             out_3x3=128,
#             out_5x5_reduce=32,
#             out_5x5=64,
#             out_7x7_reduce=32,
#             out_7x7=64,
#             dilated_3x3_reduce=32,
#             dilated_out_3x3=64,
#             pool_projection=64
#         )  # Out: 448
#         self.res1: nn.Module = ResidualBlock(channels=448)
#         self.pool1: nn.Module = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.dropout1: nn.Module = nn.Dropout(p=0.5)
#
#         # Layer 2
#         self.inception2: InceptionModule = InceptionModule(
#             in_channels=448,
#             out_1x1=128,
#             out_3x3_reduce=128,
#             out_3x3=192,
#             out_5x5_reduce=48,
#             out_5x5=96,
#             out_7x7_reduce=32,
#             out_7x7=64,
#             dilated_3x3_reduce=96,
#             dilated_out_3x3=96,
#             pool_projection=96
#         )  # Out: 672
#         self.res2: nn.Module = ResidualBlock(channels=672)
#         self.pool2: nn.Module = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.dropout2: nn.Module = nn.Dropout(p=0.5)
#
#         # Layer 3
#         self.inception3: InceptionModule = InceptionModule(
#             in_channels=672,
#             out_1x1=192,
#             out_3x3_reduce=160,
#             out_3x3=256,
#             out_5x5_reduce=64,
#             out_5x5=128,
#             out_7x7_reduce=48,
#             out_7x7=96,
#             dilated_3x3_reduce=64,
#             dilated_out_3x3=128,
#             pool_projection=128
#         )  # Out: 928
#         self.res3: nn.Module = ResidualBlock(channels=928)
#         self.dropout3: nn.Module = nn.Dropout(p=0.5)
#
#         self.avg_pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier: nn.Module = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(p=0.6),
#             nn.Linear(in_features=928, out_features=512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.6),
#             nn.Linear(in_features=512, out_features=256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.6),
#             nn.Linear(in_features=256, out_features=num_classes),
#         )
#
#         self.apply(self._initialize_weights)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.stem(x)
#
#         # Layer 1
#         x = self.inception1(x)
#         x = self.res1(x)
#         x = self.pool1(x)
#         x = self.dropout1(x)
#
#         # Layer 2
#         x = self.inception2(x)
#         x = self.res2(x)
#         x = self.pool2(x)
#         x = self.dropout2(x)
#
#         # Layer 3
#         x = self.inception3(x)
#         x = self.res3(x)
#         x = self.dropout3(x)
#
#         x = self.avg_pool(x)
#         x = self.classifier(x)
#
#         return x
#
#     def _initialize_weights(self, m):
#         """Applies Kaiming He initialization to Conv and Linear layers."""
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             # Initialize weights
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             # Initialize biases to zero
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             # Initialize BatchNorm parameters
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)

class HistoNet(nn.Module):
    def __init__(self, num_classes=3):
        super(HistoNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Layer 1
        self.inception1 = InceptionModule(96, 96, 128, 192, 32, 64, 64) # Output: 416
        self.res1 = ResidualBlock(416)
        # self.cbam1 = CBAM(416)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout(0.5)

        # Layer 2
        self.inception2 = InceptionModule(416, 192, 160, 320, 48, 128, 128) # Output: 768
        self.res2 = ResidualBlock(768)
        # self.cbam2 = CBAM(768)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout2 = nn.Dropout(0.5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(768, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes),
        )

        self.apply(self._initialize_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.res1(x)
        # x = self.cbam1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.inception2(x)
        x = self.res2(x)
        # x = self.cbam2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
