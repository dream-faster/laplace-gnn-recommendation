import pandas as pd
import numpy as np
import polars as pl


def split_data():
    print("| Loading transactions...")
    transactions = pd.read_parquet("data/original/transactions_train.parquet")

    print("| Splitting into train/val/test datasets...")
    transactions = train_test_split_by_time(transactions, "customer_id")

    print("| Saving transactions...")
    transactions.to_parquet("data/original/transactions_splitted.parquet")


def train_test_split_by_time(df: pd.DataFrame, user: str) -> pd.DataFrame:
    """This assumes data is already sorted (it is)"""
    df["train_mask"] = np.ones((len(df),), dtype=bool)
    df["val_mask"] = np.zeros((len(df),), dtype=bool)
    df["test_mask"] = np.zeros((len(df),), dtype=bool)

    def train_test_split(df):
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df

    df = df.groupby(user).apply(train_test_split).sort_index()
    return df


if __name__ == "__main__":
    split_data()
