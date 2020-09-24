from typing import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LossConfig:
    loss: str


@dataclass(frozen=True)
class KDLossConfig(LossConfig):
    alpha: float
    T: float


@dataclass(frozen=True)
class RKDLossConfig(LossConfig):
    lambda_: float
    T: float


@dataclass(frozen=True)
class LDLossConfig(LossConfig):
    alpha: float
    T: float
    is_student_training_func: Callable
