from dataclasses import dataclass


@dataclass
class Stats:
    model_id: str
    loss: list[float]
    recall_val: list[float]
    recall_test: list[float]
    precision_val: list[float]
    precision_test: list[float]
