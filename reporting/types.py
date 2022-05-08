from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseStats:
    type: Optional[str]
    epoch: Optional[int]


@dataclass
class ContinousStatsTrain(BaseStats):
    loss: float


@dataclass
class ContinousStatsVal(BaseStats):
    recall_val: float
    precision_val: float


@dataclass
class ContinousStatsTest(BaseStats):
    recall_test: float
    precision_test: float


@dataclass
class Stats:
    loss: list[float]
    recall_val: list[float]
    recall_test: list[float]
    precision_val: list[float]
    precision_test: list[float]
