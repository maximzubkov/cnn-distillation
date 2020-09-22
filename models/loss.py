import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, alpha: float, temp: float):
        super().__init__()
        self.KLDiv = nn.KLDivLoss()
        self.alpha = alpha
        self.temp = temp

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor,
                labels: torch.Tensor, is_student_evaled: bool) -> torch.Tensor:
        softmax_logits = F.log_softmax(logits / self.temp, dim=1)
        softmax_teacher_logits = F.softmax(teacher_logits / self.temp, dim=1)
        kldiv = self.KLDiv(softmax_logits, softmax_teacher_logits)
        cross_entropy = F.cross_entropy(logits, labels)
        loss = kldiv * (self.alpha * self.temp * self.temp) + cross_entropy * (1. - self.alpha)
        return loss


class AttentionLoss(nn.Module):
    def __init__(self, alpha: float, temp: float, n_cr: int, num_classes: int):
        super().__init__()
        self.attention = nn.Linear(num_classes, num_classes, bias=False)
        self.attention.weight.data.copy_(torch.eye(num_classes, num_classes))
        self.alpha = alpha
        self.temp = temp
        self.n_cr = n_cr

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor,
                labels: torch.Tensor, is_student_evaled: bool) -> torch.Tensor:
        self.attention.train() if is_student_evaled else self.attention.eval()
        softmax_logits = F.log_softmax(logits / self.temp, dim=1)
        softmax_teacher_logits = F.softmax(teacher_logits / self.temp, dim=1)
        scaling = softmax_logits.norm(p=2, dim=1) * softmax_teacher_logits.norm(p=2, dim=1)
        attn = torch.bmm(self.attention(softmax_logits).unsqueeze(1), softmax_teacher_logits.unsqueeze(-1))
        dist = attn.reshape(-1) / scaling
        loss = torch.mean(dist) * (self.alpha * self.temp * self.temp)
        cross_entropy = F.cross_entropy(logits, labels)
        loss += cross_entropy * (1. - self.alpha)
        return loss
