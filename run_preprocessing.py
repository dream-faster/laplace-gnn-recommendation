import pandas as pd
from tqdm import tqdm
from data.types import (
    DataType,
    PreprocessingConfig,
)
import torch as t
import json
import re
from run_data_splitting import train_test_split_by_time
from utils.labelencoder import encode_labels
from config import only_users_and_articles_nodes
from typing import List, Optional
from utils.preprocessing import (
    create_data_pyg,
    create_data_dgl,
    create_ids_and_maps,
    extract_edges,
    extract_reverse_edges,
)


def save_to_csv(dataframe: pd.DataFrame, name: str):
    dataframe.to_csv(f"data/saved/{name}.csv", index=False)


class PreprocessingConfig:
    filter_out_unconnected_nodes: bool
    data_size: Optional[int]
    save_to_neo4j: Optional[bool]
    data_type: DataType

    def print(self):
        print("Configuration is:")
        for key, value in vars(self).items():
            print(f"{key:>20}: {value}")


def preprocess(config: PreprocessingConfig):
    config.print()
    print("| Loading customers...")
    customers = pd.read_csv(
        "data/original/users.dat",
        delimiter="::",
        names=["customer_id", "gender", "age", "occupation", "zip"],
    )

    print("| Loading articles...")
    articles = []
    # pandas dies when is trying to parse this file, so falling back to the example code
    with open("data/original/movies.dat", encoding="latin1") as f:
        for l in f:
            id_, title, genres = l.strip().split("::")
            genres_set = set(genres.split("|"))

            # extract year
            assert re.match(r".*\([0-9]{4}\)$", title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {"article_id": int(id_), "title": title, "year": year}
            for g in genres_set:
                data[g] = 1
            articles.append(data)
    articles = pd.DataFrame(articles).fillna(0)

    print("| Loading transactions...")
    transactions = pd.read_csv(
        "data/original/ratings.dat",
        delimiter="::",
        names=["customer_id", "article_id", "rating", "timestamp"],
    )

    if config.data_size is not None:
        transactions = transactions[: config.data_size]

    print("| Encoding article features...")
    for column in tqdm(articles.columns):
        if column != "article_id":
            articles[column] = encode_labels(articles[column])

    print("| Encoding customer features...")
    for column in tqdm(customers.columns):
        if column != "customer_id":
            customers[column] = encode_labels(customers[column])

    if config.filter_out_unconnected_nodes:
        print("| Removing unconnected nodes...")
        all_article_ids_referenced = set(transactions["article_id"].unique())
        all_customer_ids_referenced = set(transactions["customer_id"].unique())
        disjoint_customers = set(customers["customer_id"].unique()).difference(
            all_customer_ids_referenced
        )
        print("|     Removing {} customers...".format(len(disjoint_customers)))
        disjoint_articles = set(articles["article_id"].unique()).difference(
            all_article_ids_referenced
        )
        print("|     Removing {} articles...".format(len(disjoint_articles)))

        customers = customers[~customers["customer_id"].isin(disjoint_customers)]
        articles = articles[~articles["article_id"].isin(disjoint_articles)]

    customers, customer_id_map_forward, customer_id_map_reverse = create_ids_and_maps(
        customers, "customer_id", 0
    )
    articles, article_id_map_forward, article_id_map_reverse = create_ids_and_maps(
        articles,
        "article_id",
        0,
    )

    print("| Parsing transactions...")
    transactions["article_id"] = transactions["article_id"].apply(
        lambda x: article_id_map_reverse[x]
    )
    transactions["customer_id"] = transactions["customer_id"].apply(
        lambda x: customer_id_map_reverse[x]
    )

    print(
        "| Splitting into train/test/val edges (using chronological stratified splitting)..."
    )
    transactions = train_test_split_by_time(transactions, "customer_id")

    transactions_train = transactions[transactions["train_mask"] == True]
    transactions_val = pd.concat(
        [transactions_train, transactions[transactions["val_mask"] == True]], axis=0
    )
    transactions_test = pd.concat(
        [transactions_val, transactions[transactions["test_mask"] == True]], axis=0
    )

    print("| Removing unused columns...")
    customers.drop(["customer_id"], axis=1, inplace=True)
    articles.drop(["article_id"], axis=1, inplace=True)

    assert config.save_to_neo4j == False, "We dont support neo4j just yet here"

    print("| Converting to tensors...")
    customers = t.tensor(customers.to_numpy(), dtype=t.long)
    assert t.isnan(customers).any() == False

    articles = t.tensor(articles.to_numpy(), dtype=t.long)
    assert t.isnan(articles).any() == False

    print("| Creating Data...")
    create_func = (
        create_data_dgl if config.data_type == DataType.dgl else create_data_pyg
    )
    train_graph = create_func(
        customers,
        articles,
        transactions_train["customer_id"].to_numpy(),
        transactions_train["article_id"].to_numpy(),
    )
    val_graph = create_func(
        customers,
        articles,
        transactions_val["customer_id"].to_numpy(),
        transactions_val["article_id"].to_numpy(),
    )
    test_graph = create_func(
        customers,
        articles,
        transactions_test["customer_id"].to_numpy(),
        transactions_test["article_id"].to_numpy(),
    )

    print("| Saving the graph...")
    t.save(train_graph, "data/derived/train_graph.pt")
    t.save(val_graph, "data/derived/val_graph.pt")
    t.save(test_graph, "data/derived/test_graph.pt")

    print("| Extracting edges per customer / per article...")
    t.save(extract_edges(transactions_train), "data/derived/edges_train.pt")
    t.save(extract_reverse_edges(transactions_train), "data/derived/rev_edges_train.pt")

    t.save(extract_edges(transactions_val), "data/derived/edges_val.pt")
    t.save(extract_reverse_edges(transactions_val), "data/derived/rev_edges_val.pt")

    t.save(extract_edges(transactions_test), "data/derived/edges_test.pt")
    t.save(extract_reverse_edges(transactions_test), "data/derived/rev_edges_test.pt")

    print("| Saving the node-to-id mapping...")
    with open("data/derived/customer_id_map_forward.json", "w") as fp:
        json.dump(customer_id_map_forward, fp)
    with open("data/derived/article_id_map_forward.json", "w") as fp:
        json.dump(article_id_map_forward, fp)


def extract_users_per_location(customers: pd.DataFrame) -> dict:
    return customers.groupby("postal_code")["index"].apply(list).to_dict()


def extract_location_for_user(customers: pd.DataFrame) -> dict:
    return customers["postal_code"].to_dict()


def read_file(filename: str) -> List[str]:
    with open(filename, encoding="latin1") as file:
        return list(file)


if __name__ == "__main__":
    preprocess(only_users_and_articles_nodes)
