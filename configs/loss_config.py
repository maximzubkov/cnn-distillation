from dataclasses import dataclass


@dataclass(frozen=True)
class LossConfig:
    loss: str


@dataclass(frozen=True)
class KDLossConfig(LossConfig):
    alpha: float
    T: float
