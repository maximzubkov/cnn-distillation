from dataclasses import dataclass


@dataclass(frozen=True)
class ModelHyperparameters:
    data_path: str

    n_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    decay_gamma: float

    shuffle_data: bool

    grad_clip: float = 0

    save_every_epoch: int = 1
    val_every_epoch: int = 1
    log_every_epoch: int = 10

    optimizer: str = "Adam"
    nesterov: bool = True
