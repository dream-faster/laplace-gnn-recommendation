import pandas as pd
from typing import List


def drop_columns_if_exist(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df = df.drop(column, axis=1)
