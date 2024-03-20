from typing import List

import torch
import torch.nn as nn


__all__ = ['Perceptron', 'MultiLayerPerceptron']


class Perceptron(nn.Module):
    def __init__(self, in_size: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size**2, num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_size: List[int], num_classes: int = 10) -> None:
        super().__init__()
        in_size.append(num_classes)
        layers = [
            Perceptron(in_size[i], in_size[i + 1] ** 2)
            for i in range(len(in_size) - 1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.layers(x)
