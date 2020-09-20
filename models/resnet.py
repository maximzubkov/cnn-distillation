from itertools import chain

import torch
import torch.nn as nn
import torchvision.models as models

from configs import ModelConfig


def _finetuning_setup(encoder: nn.Module) -> None:
    for name, child in encoder.named_children():
        if name not in ['fc']:
            for param in child.parameters():
                param.requires_grad = False
    # Unfreeze bn params
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad = True
            if hasattr(module, 'bias'):
                module.bias.requires_grad = True


class ResNet50(nn.Module):
    def __init__(self, config: ModelConfig, out_classes: int):
        super().__init__()
        self.encoder = models.resnet50(pretrained=config.pretrained)
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 1000),
            nn.Dropout(),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
        )

        if config.freeze_encoder:
            _finetuning_setup(encoder=self.encoder)

        self.classifier = nn.Linear(1000, out_classes)

        if config.is_teacher:
            children = chain(self.classifier.named_children(), self.encoder.named_children())
            for name, child in children:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class ResNet18(nn.Module):
    def __init__(self, config: ModelConfig, out_classes: int):
        super().__init__()
        self.encoder = models.resnet18(pretrained=config.pretrained)
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 1000),
            nn.Dropout(),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
        )

        if config.freeze_encoder:
            _finetuning_setup(encoder=self.encoder)

        self.classifier = nn.Linear(1000, out_classes)

        if config.is_teacher:
            children = chain(self.classifier.named_children(), self.encoder.named_children())
            for name, child in children:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)
