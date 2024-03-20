import torch
import torch.nn as nn


__all__ = ['LeNet5']


class LeNet5(nn.Module):
    def __init__(self, in_size: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        in_size = (in_size - 4) // 2
        in_size = (in_size - 4) // 2
        self.conv1 = self.__make_conv_layer(1, 6)
        self.conv2 = self.__make_conv_layer(6, 16)
        self.fc1 = self.__make_fc_layer(16*in_size**2, 120)
        self.fc2 = self.__make_fc_layer(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def __make_conv_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def __make_fc_layer(self, in_features: int, out_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
