import os
import argparse


def download_movielens():
    os.system(
        "wget -nc -P data/original/ http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    )
    os.system("unzip data/original/ml-1m.zip -d data/original/")
    os.system("rm data/original/ml-1m.zip")
    os.system("mv data/original/ml-1m/** data/original/")


def download_fashion():
    os.system(
        "wget -nc -P data/original/ https://storage.googleapis.com/heii-public/customers.parquet"
    )
    os.system(
        "wget -nc -P data/original/ https://storage.googleapis.com/heii-public/articles.parquet"
    )
    os.system(
        "wget -nc -P data/original/ https://storage.googleapis.com/heii-public/transactions_splitted.parquet"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", metavar="N", type=str)
    args = parser.parse_args()
    if args.dataset == "fashion":
        download_fashion()
    elif args.dataset == "movielens":
        download_movielens()
    else:
        raise Exception("Unknown dataset (fashion or movielens is supported)")
