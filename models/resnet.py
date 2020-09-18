import torch
import torch.nn as nn
import torchvision.models as models

from configs import ModelConfig


class ResNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = nn.Sequential(
            models.resnet18(),
            nn.Linear(1000, 10),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)
