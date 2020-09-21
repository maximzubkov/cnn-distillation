from dataclasses import dataclass

from .loss_config import LossConfig
from .model_config import ModelConfig


@dataclass(frozen=True)
class DistillationConfig:
    teacher_config: ModelConfig
    student_config: ModelConfig
    loss_config: LossConfig
