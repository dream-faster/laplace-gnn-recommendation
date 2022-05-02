import pandas as pd
import numpy as np
from utils.labelencoder import encode_labels


def split_data():
    print("| Loading transactions...")
    transactions = pd.read_parquet("data/original/transactions_train.parquet")

    print("| Deduplicating transactions...")
    transactions = deduplicate_transactions(transactions)

    print("| Splitting into train/val/test datasets...")
    transactions = train_test_split_by_time(transactions, "customer_id")

    print("| Saving transactions...")
    transactions.to_parquet("data/original/transactions_splitted.parquet")


def deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    print("|    Label encoding articles...")
    article_encoded = encode_labels(df["article_id"]).astype(str)
    print("|    Label encoding customers...")
    customer_encoded = encode_labels(df["customer_id"]).astype(str)
    print("|    Concatenating customer-articles...")
    df["customer-article"] = pd.concat([customer_encoded, article_encoded], axis=1).agg(
        "-".join
    )
    print("|    Drop duplicates...")
    df.drop_duplicates(subset=["customer-article"], keep="first", inplace=True)
    df.drop(columns=["customer-article"], inplace=True)
    return df


# This is the train-test split method most of the recommender system papers running on MovieLens
# takes.  It essentially follows the intuition of "training on the past and predict the future".
# One can also change the threshold to make validation and test set take larger proportions.
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
