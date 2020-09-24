from typing import Dict

import torch
from pytorch_lightning.metrics.functional import confusion_matrix

from configs import DistillationConfig, ModelHyperparameters
from .base_cifar_model import BaseCifarModel
from .loss import KDLoss, RKDDistanceLoss, RKDAngleLoss


class DistillationCifarModel(BaseCifarModel):
    def __init__(self, model_config: DistillationConfig,
                 hyperparams_config: ModelHyperparameters, num_workers: int = 0):
        super().__init__(hyperparams_config, num_workers)
        self.loss_config = model_config.loss_config
        if self.loss_config.loss == "KD":
            self.criterion = KDLoss(alpha=self.loss_config.alpha,
                                    temp=self.loss_config.T)
        elif self.loss_config.loss == "RKD_Dist":
            self.criterion = RKDDistanceLoss(lambda_=self.loss_config.lambda_)
        elif self.loss_config.loss == "RKD_Angle":
            self.criterion = RKDAngleLoss(lambda_=self.loss_config.lambda_)
        else:
            raise ValueError(f"Unknown loss function {self.loss_config.loss}")
        self.student = self.get_model(model_config.student_config)
        self.teacher = self.get_model(model_config.teacher_config)
        self.save_hyperparameters()

    def forward(self, images: torch.Tensor):
        return self.student(images)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, labels = batch
        student_features = self.criterion.extract_features(self.student, batch)
        logits = student_features.logits
        with torch.no_grad():
            teacher_features = self.criterion.extract_features(self.teacher, batch)
        loss = self.criterion(student_features=student_features, teacher_features=teacher_features, labels=labels)
        with torch.no_grad():
            log = {'train/loss': loss}
            conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))
            log["train/accuracy"] = conf_matrix.trace() / conf_matrix.sum()
        progress_bar = {"train/accuracy": log["train/accuracy"]}

        return {"loss": loss, "log": log, "progress_bar": progress_bar, "confusion_matrix": conf_matrix}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        # [batch size; num_classes]
        images, labels = batch
        student_features = self.criterion.extract_features(self.student, batch)
        teacher_features = self.criterion.extract_features(self.teacher, batch)
        loss = self.criterion(student_features=student_features, teacher_features=teacher_features, labels=labels)
        logits = student_features.logits
        conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))

        return {"val_loss": loss, "confusion_matrix": conf_matrix}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result
