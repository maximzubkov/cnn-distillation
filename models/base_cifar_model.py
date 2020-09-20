from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as transforms
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import confusion_matrix
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from configs import ModelConfig, ModelHyperparameters
from .resnet import ResNet18, ResNet50


class BaseCifarModel(LightningModule):
    def __init__(self, hyperparams_config: ModelHyperparameters, num_workers: int = 0):
        super().__init__()
        self.hyperparams = hyperparams_config
        self.num_workers = num_workers
        self.num_classes = 10
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def get_model(self, model_config: ModelConfig) -> torch.nn:
        if model_config.model_name == "resnet":
            if model_config.num_layers == 50:
                model = ResNet50(model_config, self.num_classes)
            elif model_config.num_layers == 18:
                model = ResNet18(model_config, self.num_classes)
            else:
                raise ValueError("Unknown resnet")
        else:
            raise ValueError("Unknown model")
        if model_config.is_teacher:
            checkpoints = torch.load(model_config.checkpoint_path, map_location=self.device)
            parsed_state_dict = OrderedDict()
            for k, v in checkpoints["state_dict"].items():
                name = k[6:]  # remove `module.`
                parsed_state_dict[name] = v
            model.load_state_dict(parsed_state_dict)
        return model

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

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.Resize((128, 128)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hyperparams.data_path,
                          train=True,
                          download=True,
                          transform=transform_train)
        return DataLoader(dataset,
                          batch_size=self.hyperparams.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.hyperparams.shuffle_data,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.Resize((128, 128)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hyperparams.data_path,
                          train=False,
                          download=True,
                          transform=transform_val)
        return DataLoader(dataset,
                          batch_size=self.hyperparams.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()

    # ===== ON EPOCH END =====

    def _general_epoch_end(self, outputs: List[Dict], loss_key: str, group: str) -> Dict:
        with torch.no_grad():
            logs = {f"{group}/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
            accumulated_conf_matrix = torch.zeros(
                self.num_classes, self.num_classes, requires_grad=False, device=self.device
            )
            for out in outputs:
                _conf_matrix = out["confusion_matrix"]
                max_class_index, _ = _conf_matrix.shape
                accumulated_conf_matrix[:max_class_index, :max_class_index] += _conf_matrix
            logs[f"{group}/accuracy"] = accumulated_conf_matrix.trace() / accumulated_conf_matrix.sum()
        progress_bar = {k: v for k, v in logs.items() if k in [f"{group}/loss", f"{group}/accuracy"]}
        return {"val_loss": logs[f"{group}/loss"], "log": logs, "progress_bar": progress_bar}

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "loss", "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val_loss", "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test_loss", "test")
