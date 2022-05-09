from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseStats:
    type: Optional[str]


@dataclass
class ContinousStatsTrain(BaseStats):
    loss: float
    epoch: int


@dataclass
class ContinousStatsVal(BaseStats):
    recall_val: float
    precision_val: float
    epoch: int


@dataclass
class ContinousStatsTest(BaseStats):
    recall_test: float
    precision_test: float


@dataclass
class Stats:
    loss: float
    recall_val: float
    recall_test: float
    precision_val: float
    precision_test: float
