from typing import Dict
from random import random

import torch
from pytorch_lightning.metrics.functional import confusion_matrix

from configs import DistillationConfig, ModelHyperparameters
from .base_cifar_model import BaseCifarModel
from .loss import KDLoss, AttentionLoss


class DistillationCifarModel(BaseCifarModel):
    def __init__(self, model_config: DistillationConfig,
                 hyperparams_config: ModelHyperparameters, num_workers: int = 0):
        super().__init__(hyperparams_config, num_workers)
        self.loss_config = model_config.loss_config
        if self.loss_config.loss == "KD":
            self.criterion = KDLoss(alpha=self.loss_config.alpha,
                                    temp=self.loss_config.T)
            self.is_student_eval_func = lambda batch_idx: False
        elif self.loss_config.loss == "Attention":
            self.criterion = AttentionLoss(alpha=self.loss_config.alpha,
                                           temp=self.loss_config.T)
            self.is_student_eval_func = lambda: random() < self.loss_config.p
        else:
            raise ValueError(f"Unknown loss function {self.loss_config.loss}")
        self.student = self.get_model(model_config.student_config)
        self.teacher = self.get_model(model_config.teacher_config)
        self.save_hyperparameters()

    def forward(self, images: torch.Tensor):
        return self.student(images)

    def _compute_loss(self, logits: torch.Tensor, batch: torch.Tensor, is_student_evaled: bool) -> torch.Tensor:
        images, labels = batch
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        return self.criterion(logits, teacher_logits, labels, is_student_evaled)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        self.student.train()
        images, labels = batch
        is_student_evaled = self.is_student_eval_func()
        self.student.eval() if is_student_evaled else self.student.train()
        logits = self.student(images)
        loss = self._compute_loss(logits, batch, is_student_evaled)
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
        loss = self._compute_loss(logits, batch, is_student_evaled=False)
        conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))

        return {"val_loss": loss, "confusion_matrix": conf_matrix}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result
