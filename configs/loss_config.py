from dataclasses import dataclass


@dataclass(frozen=True)
class LossConfig:
    loss: str


@dataclass(frozen=True)
class KDLossConfig(LossConfig):
    alpha: float
    T: float


@dataclass(frozen=True)
class AttentionLossConfig(LossConfig):
    alpha: float
    T: float
    eps: float
    n_cr: int
