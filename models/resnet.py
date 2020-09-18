from abc import abstractmethod
from typing import Tuple, Dict, List

from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.utils.data import DataLoader

from configs import ResNetConfig


class ResNet(LightningModule):
    def __init__(self, model_config: ResNetConfig, num_workers: int = 0):
        super().__init__()
        self.hyperparams = model_config.hyperparams_config
        self.model_config = model_config
        self.num_workers = num_workers

    @abstractmethod
    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        pass

    # ===== OPTIMIZERS =====

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        if self.hyperparams.optimizer == "Momentum":
            # using the same momentum value as in original realization by Alon
            optimizer = SGD(
                self.parameters(),
                self.hyperparams.learning_rate,
                momentum=0.95,
                nesterov=self.hyperparams.nesterov,
                weight_decay=self.hyperparams.weight_decay,
            )
        elif self.hyperparams.optimizer == "Adam":
            optimizer = Adam(
                self.parameters(),
                self.hyperparams.learning_rate,
                weight_decay=self.hyperparams.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.hyperparams.optimizer}, try one of: Adam, Momentum")
        scheduler = ExponentialLR(optimizer, self.hyperparams.decay_gamma)
        return [optimizer], [scheduler]

    # ===== DATALOADERS BLOCK =====

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass

    # ===== ON EPOCH END =====

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "loss", "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val_loss", "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test_loss", "test")
