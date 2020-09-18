from dataclasses import dataclass

from .hyperparams_config import ModelHyperparameters


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    hyperparams_config: ModelHyperparameters
