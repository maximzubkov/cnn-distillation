from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KDFeatures:
    logits: torch.Tensor


class KDLoss(nn.Module):
    def __init__(self, alpha: float, temp: float):
        super().__init__()
        self.KLDiv = nn.KLDivLoss()
        self.alpha = alpha
        self.temp = temp

    @staticmethod
    def extract_features(model: nn.Module, batch: torch.Tensor) -> KDFeatures:
        images, _ = batch
        logits = model(images)
        return KDFeatures(logits)

    def forward(self, student_features: KDFeatures, teacher_features: KDFeatures,
                labels: torch.Tensor) -> torch.Tensor:
        student_logits = student_features.logits
        teacher_logits = teacher_features.logits
        soft_student_logits = F.log_softmax(student_logits / self.temp, dim=1)
        soft_teacher_logits = F.softmax(teacher_logits / self.temp, dim=1)
        kldiv = self.KLDiv(soft_student_logits, soft_teacher_logits)
        cross_entropy = F.cross_entropy(student_logits, labels)
        loss = kldiv * (self.alpha * self.temp * self.temp) + cross_entropy * (1. - self.alpha)
        return loss
