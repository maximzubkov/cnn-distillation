from typing import Dict

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import confusion_matrix

from configs import ModelConfig, ModelHyperparameters
from .base_cifar_model import BaseCifarModel, get_model


class SingleCifarModel(BaseCifarModel):
    def __init__(self, model_config: ModelConfig,
                 hyperparams_config: ModelHyperparameters, num_workers: int = 0):
        super().__init__(hyperparams_config, num_workers)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = get_model(model_config)

    def forward(self, images: torch.Tensor):
        return self.model(images)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        log = {'train/loss': loss}
        with torch.no_grad():
            conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))
            log["train/accuracy"] = conf_matrix.trace() / conf_matrix.sum()
        progress_bar = {"train/accuracy": log["train/accuracy"]}

        return {"loss": loss, "log": log, "progress_bar": progress_bar, "confusion_matrix": conf_matrix}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        # [batch size; num_classes]
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels.squeeze(0))
        conf_matrix = confusion_matrix(logits.argmax(-1), labels.squeeze(0))

        return {"val_loss": loss, "confusion_matrix": conf_matrix}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result
