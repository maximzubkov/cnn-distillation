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


class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=4, stride=1)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=1)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        out = self.conv1(input_)
        out = self.relu1(self.bn1(out))
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        out = self.pool2(out)
        out = self.sigmoid(self.linear(out.squeeze()))
        return out


class AttentionLoss(nn.Module):
    def __init__(self, alpha: float, temp: float, n_cr: int, num_classes: int):
        super().__init__()
        self.discriminator = TinyConv()
        self.alpha = alpha
        self.temp = temp
        self.n_cr = n_cr

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor,
                labels: torch.Tensor, is_student_evaled: bool) -> torch.Tensor:
        softmax_logits = F.log_softmax(logits / self.temp, dim=1)
        softmax_teacher_logits = F.softmax(teacher_logits / self.temp, dim=1)
        if is_student_evaled:
            self.discriminator.train()
            loss = self.discriminator_loss(softmax_logits, softmax_teacher_logits)
        else:
            self.discriminator.eval()
            loss = self.generator_loss(softmax_logits)
        loss *= (self.alpha * self.temp * self.temp)
        loss += F.cross_entropy(logits, labels) * (1. - self.alpha)
        return loss

    def generator_loss(self, student_logits: torch.Tensor) -> torch.Tensor:
        """
        Wassersten loss for student
        """
        return -self.discriminator(student_logits.unsqueeze(1)).mean()

    def discriminator_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Wassersten loss for discriminator
        """
        s_loss = self.discriminator(student_logits.unsqueeze(1)).mean()
        t_loss = self.discriminator(teacher_logits.unsqueeze(1)).mean()
        return s_loss - t_loss
