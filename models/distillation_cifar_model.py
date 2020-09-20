from typing import Dict

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import confusion_matrix

from configs import DistillationConfig, ModelHyperparameters
from .base_cifar_model import BaseCifarModel


class DistillationCifarModel(BaseCifarModel):
    def __init__(self, model_config: DistillationConfig,
                 hyperparams_config: ModelHyperparameters, num_workers: int = 0):
        super().__init__(hyperparams_config, num_workers)
        self.criterion = torch.nn.MSELoss()
        self.student = self.get_model(model_config.student_config)
        self.teacher = self.get_model(model_config.teacher_config)
        self.save_hyperparameters()

    def forward(self, images: torch.Tensor):
        return self.model(images)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, labels = batch
        student_encoding = self.student.encode(images)
        teacher_encoding = self.teacher.encode(images)
        loss = self.criterion(teacher_encoding, student_encoding)
        logits = self.student.classifier(student_encoding)
        log = {'train/loss': loss}
        with torch.no_grad():
            conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))
            log["train/accuracy"] = conf_matrix.trace() / conf_matrix.sum()
        progress_bar = {"train/accuracy": log["train/accuracy"]}

        return {"loss": loss, "log": log, "progress_bar": progress_bar, "confusion_matrix": conf_matrix}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        # [batch size; num_classes]
        images, labels = batch
        student_encoding = self.student.encode(images)
        teacher_encoding = self.teacher.encode(images)
        loss = F.mse_loss(student_encoding, teacher_encoding)
        logits = self.student.classifier(student_encoding)
        conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))

        return {"val_loss": loss, "confusion_matrix": conf_matrix}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result
