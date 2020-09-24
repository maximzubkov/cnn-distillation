from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pdist


@dataclass(frozen=True)
class RKDFeatures:
    logits: torch.Tensor


class RKDAngleLoss(nn.Module):
    def __init__(self, lambda_: float, temp: float):
        super().__init__()
        self.lambda_ = lambda_
        self.temp = temp

    @staticmethod
    def extract_features(model: nn.Module, batch: torch.Tensor) -> RKDFeatures:
        images, _ = batch
        logits = model(images)
        return RKDFeatures(logits)

    def forward(self, student_features: RKDFeatures, teacher_features: RKDFeatures,
                labels: torch.Tensor) -> torch.Tensor:
        student_logits = student_features.logits
        teacher_logits = teacher_features.logits
        soft_student_logits = F.log_softmax(student_logits / self.temp, dim=1)
        soft_teacher_logits = F.softmax(teacher_logits / self.temp, dim=1)

        with torch.no_grad():
            td = (soft_teacher_logits.unsqueeze(0) - soft_teacher_logits.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (soft_student_logits.unsqueeze(0) - student_logits.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        rkd = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        cross_entropy = F.cross_entropy(student_logits, labels)
        loss = rkd * self.lambda_ + cross_entropy
        return loss


class RKDDistanceLoss(nn.Module):
    def __init__(self, lambda_: float, temp: float):
        super().__init__()
        self.lambda_ = lambda_
        self.temp = temp

    @staticmethod
    def extract_features(model: nn.Module, batch: torch.Tensor) -> RKDFeatures:
        images, _ = batch
        logits = model(images)
        return RKDFeatures(logits)

    def forward(self, student_features: RKDFeatures, teacher_features: RKDFeatures,
                labels: torch.Tensor) -> torch.Tensor:
        student_logits = student_features.logits
        teacher_logits = teacher_features.logits
        soft_student_logits = F.log_softmax(student_logits / self.temp, dim=1)
        soft_teacher_logits = F.softmax(teacher_logits / self.temp, dim=1)

        with torch.no_grad():
            t_d = pdist(soft_teacher_logits, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(soft_student_logits, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        rkd = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        cross_entropy = F.cross_entropy(student_logits, labels)
        loss = rkd * self.lambda_ + cross_entropy
        return loss
