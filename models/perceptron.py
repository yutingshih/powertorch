import torch
import torch.nn as nn


__all__ = ['Perceptron']


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
