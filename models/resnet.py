from itertools import chain

import torch
import torch.nn as nn
import torchvision.models as models

from configs import ModelConfig


class ResNet50(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = models.resnet50(pretrained=config.pretrained)
        if config.freeze_encoder:
            for name, child in self.encoder.named_children():
                for param in child.parameters():
                    param.requires_grad = False

        self.classifier = nn.Linear(1000, 10)
        if config.is_teacher:
            childern = chain(self.classifier.named_children(), self.encoder.named_children())
            for name, child in childern:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class ResNet18(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = models.resnet18(pretrained=config.pretrained)
        if config.freeze_encoder:
            for name, child in self.encoder.named_children():
                for param in child.parameters():
                    param.requires_grad = False

        self.classifier = nn.Linear(1000, 10)
        if config.is_teacher:
            childern = chain(self.classifier.named_children(), self.encoder.named_children())
            for name, child in childern:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)
