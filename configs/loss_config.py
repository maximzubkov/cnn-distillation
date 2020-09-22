from dataclasses import dataclass


@dataclass(frozen=True)
class LossConfig:
    loss: str


@dataclass(frozen=True)
class KDLossConfig(LossConfig):
    alpha: float
    T: float


@dataclass(frozen=True)
class SinkhornLossConfig(LossConfig):
    alpha: float
    T: float
    eps: float
    max_iter: int
