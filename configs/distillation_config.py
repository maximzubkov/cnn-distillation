from dataclasses import dataclass

from .model_config import ModelConfig
from .loss_config import LossConfig


@dataclass(frozen=True)
class DistillationConfig:
    teacher_config: ModelConfig
    student_config: ModelConfig
    loss_config: LossConfig
