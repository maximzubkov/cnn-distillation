from dataclasses import dataclass

from .model_config import ModelConfig


@dataclass(frozen=True)
class DistillationConfig:
    teacher_config: ModelConfig
    student_config: ModelConfig
