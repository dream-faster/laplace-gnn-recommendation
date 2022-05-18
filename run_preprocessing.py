import pandas as pd
from tqdm import tqdm
from data.types import (
    DataType,
    PreprocessingConfig,
    UserColumn,
)
import torch as t
import json
from utils.labelencoder import encode_labels
import numpy as np
from typing import Tuple
from config import only_users_and_articles_nodes
import numpy as np
from utils.constants import Constants
import os
import time


def save_to_csv(dataframe: pd.DataFrame, name: str):
    dataframe.to_csv(f"data/saved/{name}.csv", index=False)


def preprocess(config: PreprocessingConfig):
    config.print()
    print("| Loading customers...")
    customers = pd.read_parquet("data/original/customers.parquet").fillna(0.0)
    customers = customers[[c.value for c in config.customer_features] + ["customer_id"]]

    print("| Loading articles...")
    articles = pd.read_parquet("data/original/articles.parquet").fillna(0.0)

    print("| Loading transactions...")
    transactions = pd.read_parquet("data/original/transactions_splitted.parquet")
    if config.data_size is not None:
        transactions = transactions[: config.data_size]

    transactions["year-month"] = pd.to_datetime(transactions["t_dat"]).dt.strftime(
        "%Y-%m"
    )

    print("| Calculating average price per product...")
    transactions_per_article = transactions.groupby(["article_id"]).mean()["price"]
    articles = articles.merge(
        transactions_per_article, on="article_id", how="outer"
    ).fillna(0.0)

    articles = articles[[c.value for c in config.article_features] + ["article_id"]]

    print("| Encoding article features...")
    for column in tqdm(articles.columns):
        if (
            column not in config.article_non_categorical_features
            and column != "article_id"
        ):
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
    transactions_train = transactions[transactions["train_mask"] == True]
    transactions_val = pd.concat(
        [transactions_train, transactions[transactions["val_mask"] == True]], axis=0
    )
    transactions_test = pd.concat(
        [transactions_val, transactions[transactions["test_mask"] == True]], axis=0
    )

    per_article_img_embedding = t.zeros((0, 512))
    if config.load_image_embedding:
        print("| Adding image embeddings...")
        image_embeddings = t.load(
            "data/derived/fashion-recommendation-image-embeddings-clip-ViT-B-32.pt"
        )
        for index, article in tqdm(articles.iterrows()):
            per_article_img_embedding = t.cat(
                (
                    per_article_img_embedding,
                    image_embeddings.get(
                        int(article["article_id"]), t.zeros(512)
                    ).unsqueeze(0),
                ),
                axis=0,
            )

    per_article_text_embedding = t.zeros((0, 512))
    if config.load_text_embedding:
        print("| Adding text embeddings...")
        text_embeddings = t.load(
            "data/derived/fashion-recommendation-text-embeddings-clip-ViT-B-32.pt"
        )

        for index, article in tqdm(articles.iterrows()):
            per_article_text_embedding = t.cat(
                (
                    per_article_text_embedding,
                    text_embeddings[int(article["article_id"])]
                    .get(config.text_embedding_colname, t.zeros(512))
                    .unsqueeze(0),
                ),
                axis=0,
            )

    print("| Exporting per location info...")
    t.save(
        extract_users_per_location(customers), "data/derived/customers_per_location.pt"
    )
    t.save(extract_location_for_user(customers), "data/derived/location_for_user.pt")

    print("| Calculating the most popular products of the month...")
    print("last day:", transactions.tail(1)["t_dat"].item())
    last_month = transactions.tail(1)["year-month"].item()
    last_month_transactions = transactions[transactions["year-month"] == last_month]
    most_popular_products = (
        last_month_transactions["article_id"].value_counts().nlargest(1000)
    )
    t.save(most_popular_products, "data/derived/most_popular_products.pt")

    print("| Removing unused columns...")
    customers.drop(["customer_id"], axis=1, inplace=True)
    articles.drop(["article_id"], axis=1, inplace=True)

    if config.save_to_neo4j:
        save_to_neo4j(customers, articles, transactions)

    print("| Converting to tensors...")
    customers = t.tensor(customers.to_numpy(), dtype=t.float)
    assert t.isnan(customers).any() == False

    articles = t.tensor(articles.to_numpy(), dtype=t.float)
    if config.load_image_embedding:
        articles = t.cat((articles, per_article_img_embedding), axis=1)
    if config.load_text_embedding:
        articles = t.cat((articles, per_article_text_embedding), axis=1)
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


def create_data_pyg(
    customers: t.Tensor,
    articles: t.Tensor,
    transactions_to_customer_id: np.ndarray,
    transactions_to_article_id: np.ndarray,
):

    from torch_geometric.data import HeteroData

    data = HeteroData()
    data[Constants.node_user].x = customers
    data[Constants.node_item].x = articles
    data[Constants.node_user, "buys", Constants.node_item].edge_index = t.as_tensor(
        (transactions_to_customer_id, transactions_to_article_id),
        dtype=t.long,
    )
    return data


def create_data_dgl(
    customers: t.Tensor,
    articles: t.Tensor,
    transactions_to_customer_id: np.ndarray,
    transactions_to_article_id: np.ndarray,
):

    import dgl

    data = dgl.heterograph(
        {
            Constants.edge_key: (
                t.as_tensor(transactions_to_customer_id, dtype=t.long),
                t.as_tensor(transactions_to_article_id, dtype=t.long),
            ),
            (Constants.node_item, "rev_buys", Constants.node_user): (
                t.as_tensor(transactions_to_article_id, dtype=t.long),
                t.as_tensor(transactions_to_customer_id, dtype=t.long),
            ),
        },
        num_nodes_dict={
            Constants.node_user: customers.shape[0],
            Constants.node_item: articles.shape[0],
        },
    )
    data.nodes[Constants.node_user].data["features"] = customers
    data.nodes[Constants.node_item].data["features"] = articles
    return data


def create_ids_and_maps(
    df: pd.DataFrame, column: str, start: int
) -> Tuple[pd.DataFrame, dict, dict]:
    df.reset_index(inplace=True)
    df.index += start
    mapping_forward = df[column].to_dict()
    mapping_reverse = {v: k for k, v in mapping_forward.items()}
    df["index"] = df.index
    return df, mapping_forward, mapping_reverse


def extract_edges(transactions: pd.DataFrame) -> dict:
    return transactions.groupby("customer_id")["article_id"].apply(list).to_dict()


def extract_reverse_edges(transactions: pd.DataFrame) -> dict:
    return transactions.groupby("article_id")["customer_id"].apply(list).to_dict()


def extract_users_per_location(customers: pd.DataFrame) -> dict:
    return customers.groupby("postal_code")["index"].apply(list).to_dict()


def extract_location_for_user(customers: pd.DataFrame) -> dict:
    return customers["postal_code"].to_dict()


def save_to_neo4j(
    customers: pd.DataFrame, articles: pd.DataFrame, transactions: pd.DataFrame
):
    print("| Saving to neo4j...")
    customers = customers.copy()
    customers[":LABEL"] = "Customer"
    customers.rename(columns={"index": ":ID(Customer)"}, inplace=True)
    customers["_id"] = customers[":ID(Customer)"]
    save_to_csv(customers, "customers")

    articles = articles.copy()
    articles[":LABEL"] = "Article"
    articles.rename(columns={"index": ":ID(Article)"}, inplace=True)
    articles["_id"] = articles[":ID(Article)"]
    save_to_csv(articles, "articles")

    transactions = transactions.copy()
    transactions.rename(
        columns={
            "customer_id": ":START_ID(Customer)",
            "article_id": ":END_ID(Article)",
        },
        inplace=True,
    )
    transactions["train_mask"] = transactions["train_mask"].astype(int)
    transactions["test_mask"] = transactions["test_mask"].astype(int)
    transactions["val_mask"] = transactions["val_mask"].astype(int)
    transactions.drop(
        ["t_dat", "price", "sales_channel_id", "year-month"], axis=1, inplace=True
    )
    transactions[":TYPE"] = "BUYS"
    save_to_csv(transactions, "transactions")
    # Neo4j needs to be stopped for neo4j-admin import to run
    print("| Stopping running instances of Neo4j...")
    os.system("neo4j stop")
    print("| Importing csv to database...")
    os.system(
        "neo4j-admin import --database=neo4j --nodes=data/saved/articles.csv --nodes=data/saved/customers.csv --relationships=data/saved/transactions.csv --force"
    )
    print("| Starting Neo4j...")
    os.system("neo4j start")
    time.sleep(5)
    # Create the indexes for Customer & Article node types
    print("| Creating indexes...")

    os.system(
        "echo 'CREATE INDEX ON :Customer(ID)' | cypher-shell -u neo4j -p password --format plain"
    )
    os.system(
        "echo 'CREATE INDEX ON :Customer(_id)' | cypher-shell -u neo4j -p password --format plain"
    )
    os.system(
        "echo 'CREATE INDEX ON :Article(ID)' | cypher-shell -u neo4j -p password --format plain"
    )
    os.system(
        "echo 'CREATE INDEX ON :Article(_id)' | cypher-shell -u neo4j -p password --format plain"

    print("Number of nodes in the database:")
    os.system(
        "echo 'MATCH (n) RETURN count(n)' | cypher-shell -u neo4j -p password --format plain"

    )


if __name__ == "__main__":
    preprocess(only_users_and_articles_nodes)
