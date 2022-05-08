import pandas as pd
from tqdm import tqdm
from data.types import (
    DataType,
    PreprocessingConfig,
    UserColumn,
)
import torch
import json
from utils.labelencoder import encode_labels
import numpy as np
from typing import Tuple
from config import only_users_and_articles_nodes
import numpy as np
from utils.constants import Constants


def save_to_csv(
    customers: pd.DataFrame, articles: pd.DataFrame, transactions: pd.DataFrame
):
    customers.to_csv("data/saved/customers.csv", index=False)
    articles.to_csv("data/saved/articles.csv", index=False)
    transactions.to_csv("data/saved/transactions.csv", index=False)


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

    per_article_img_embedding = torch.zeros((0, 512))
    if config.load_image_embedding:
        print("| Adding image embeddings...")
        image_embeddings = torch.load(
            "data/derived/fashion-recommendation-image-embeddings-clip-ViT-B-32.pt"
        )
        for index, article in tqdm(articles.iterrows()):
            per_article_img_embedding = torch.cat(
                (
                    per_article_img_embedding,
                    image_embeddings.get(
                        int(article["article_id"]), torch.zeros(512)
                    ).unsqueeze(0),
                ),
                axis=0,
            )

    per_article_text_embedding = torch.zeros((0, 512))
    if config.load_text_embedding:
        print("| Adding text embeddings...")
        text_embeddings = torch.load(
            "data/derived/fashion-recommendation-text-embeddings-clip-ViT-B-32.pt"
        )

        for index, article in tqdm(articles.iterrows()):
            per_article_text_embedding = torch.cat(
                (
                    per_article_text_embedding,
                    text_embeddings[int(article["article_id"])]
                    .get(config.text_embedding_colname, torch.zeros(512))
                    .unsqueeze(0),
                ),
                axis=0,
            )

    print("| Exporting per location info...")
    torch.save(
        extract_users_per_location(customers), "data/derived/customers_per_location.pt"
    )
    torch.save(
        extract_location_for_user(customers), "data/derived/location_for_user.pt"
    )

    print("| Calculating the most popular products of the month...")
    print("last day:", transactions.tail(1)["t_dat"].item())
    last_month = transactions.tail(1)["year-month"].item()
    last_month_transactions = transactions[transactions["year-month"] == last_month]
    most_popular_products = (
        last_month_transactions["article_id"].value_counts().nlargest(1000)
    )
    torch.save(most_popular_products, "data/derived/most_popular_products.pt")

    print("| Removing unused columns...")
    customers.drop(["customer_id"], axis=1, inplace=True)
    articles.drop(["article_id"], axis=1, inplace=True)

    if config.save_to_csv:
        save_to_csv(customers, articles, transactions)

    print("| Converting to tensors...")
    customers = torch.tensor(customers.to_numpy(), dtype=torch.float)
    assert torch.isnan(customers).any() == False

    articles = torch.tensor(articles.to_numpy(), dtype=torch.float)
    if config.load_image_embedding:
        articles = torch.cat((articles, per_article_img_embedding), axis=1)
    if config.load_text_embedding:
        articles = torch.cat((articles, per_article_text_embedding), axis=1)
    assert torch.isnan(articles).any() == False

    print("| Creating PyG Data...")
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
    torch.save(train_graph, "data/derived/train_graph.pt")
    torch.save(val_graph, "data/derived/val_graph.pt")
    torch.save(test_graph, "data/derived/test_graph.pt")

    print("| Extracting edges per customer / per article...")
    torch.save(extract_edges(transactions_train), "data/derived/edges_train.pt")
    torch.save(
        extract_reverse_edges(transactions_train), "data/derived/rev_edges_train.pt"
    )
    torch.save(
        extract_edges_articles(transactions_train),
        "data/derived/articles_edges_train.pt",
    )

    torch.save(extract_edges(transactions_val), "data/derived/edges_val.pt")
    torch.save(extract_reverse_edges(transactions_val), "data/derived/rev_edges_val.pt")
    torch.save(
        extract_edges_articles(transactions_val), "data/derived/articles_edges_val.pt"
    )

    torch.save(extract_edges(transactions_test), "data/derived/edges_test.pt")
    torch.save(
        extract_reverse_edges(transactions_test), "data/derived/rev_edges_test.pt"
    )
    torch.save(
        extract_edges_articles(transactions_test), "data/derived/articles_edges_test.pt"
    )

    print("| Saving the node-to-id mapping...")
    with open("data/derived/customer_id_map_forward.json", "w") as fp:
        json.dump(customer_id_map_forward, fp)
    with open("data/derived/article_id_map_forward.json", "w") as fp:
        json.dump(article_id_map_forward, fp)


def create_data_pyg(
    customers: torch.Tensor,
    articles: torch.Tensor,
    transactions_to_customer_id: np.ndarray,
    transactions_to_article_id: np.ndarray,
):

    from torch_geometric.data import HeteroData

    data = HeteroData()
    data[Constants.node_user].x = customers
    data[Constants.node_item].x = articles
    data[Constants.node_user, "buys", Constants.node_item].edge_index = torch.as_tensor(
        (transactions_to_customer_id, transactions_to_article_id),
        dtype=torch.long,
    )
    return data


def create_data_dgl(
    customers: torch.Tensor,
    articles: torch.Tensor,
    transactions_to_customer_id: np.ndarray,
    transactions_to_article_id: np.ndarray,
):

    import dgl

    data = dgl.heterograph(
        {
            Constants.edge_key: (
                torch.as_tensor(transactions_to_customer_id, dtype=torch.long),
                torch.as_tensor(transactions_to_article_id, dtype=torch.long),
            ),
            (Constants.node_item, "rev_buys", Constants.node_user): (
                torch.as_tensor(transactions_to_article_id, dtype=torch.long),
                torch.as_tensor(transactions_to_customer_id, dtype=torch.long),
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


def extract_edges_articles(transactions: pd.DataFrame) -> dict:
    return transactions.groupby("article_id")["customer_id"].apply(list).to_dict()


def extract_users_per_location(customers: pd.DataFrame) -> dict:
    return customers.groupby("postal_code")["index"].apply(list).to_dict()


def extract_location_for_user(customers: pd.DataFrame) -> dict:
    return customers["postal_code"].to_dict()


if __name__ == "__main__":
    preprocess(only_users_and_articles_nodes)
