from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning.metrics.functional import confusion_matrix

from configs import DistillationConfig, ModelHyperparameters
from .base_cifar_model import BaseCifarModel


class LogitsDiscriminatorCifarModel(BaseCifarModel):
    def __init__(self, model_config: DistillationConfig,
                 hyperparams_config: ModelHyperparameters, num_workers: int = 0):
        super().__init__(hyperparams_config, num_workers)
        self.loss_config = model_config.loss_config
        self.student = self.get_model(model_config.student_config)
        self.teacher = self.get_model(model_config.teacher_config)
        self.discriminator = TinyConvDiscriminator()
        self.save_hyperparameters()

    def forward(self, images: torch.Tensor):
        return self.student(images)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        self.student.train()
        images, labels = batch
        is_student_evaled = self.is_student_eval_func()
        self.student.eval() if is_student_evaled else self.student.train()
        logits = self.student(images)
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        loss = self.criterion(logits, teacher_logits, labels, is_student_evaled)
        if is_student_evaled:
            self.student.zero_grad()
        with torch.no_grad():
            log = {'train/loss': loss}
            conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))
            log["train/accuracy"] = conf_matrix.trace() / conf_matrix.sum()
        progress_bar = {"train/accuracy": log["train/accuracy"]}

        return {"loss": loss, "log": log, "progress_bar": progress_bar, "confusion_matrix": conf_matrix}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        # [batch size; num_classes]
        images, labels = batch
        logits = self.student(images)
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        loss = self.criterion(logits, teacher_logits, labels, is_student_evaled=False)
        conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))

        return {"val_loss": loss, "confusion_matrix": conf_matrix}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result


class TinyConvDiscriminator(nn.Module):
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
