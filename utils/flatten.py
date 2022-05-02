from typing import List

def flatten(t: List) -> List:
    return [item for sublist in t for item in sublist]