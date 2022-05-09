from dataclasses import dataclass
from typing import Optional, List, Union


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
    loss: List[float]
    recall_val: List[float]
    recall_test: List[float]
    precision_val: List[float]
    precision_test: List[float]
