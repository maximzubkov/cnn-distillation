from dataclasses import dataclass


@dataclass(frozen=True)
class LossConfig:
    loss: str


@dataclass(frozen=True)
class KDLossConfig(LossConfig):
    alpha: float
    temp: float


@dataclass(frozen=True)
class RKDLossConfig(LossConfig):
    lambda_: float
    temp: float
