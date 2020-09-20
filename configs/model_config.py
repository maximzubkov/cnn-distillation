from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    num_layers: int
    pretrained: bool
    is_teacher: bool
    freeze_encoder: bool
    checkpoint_path: str = None
