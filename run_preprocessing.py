import pandas as pd
from tqdm import tqdm
from data.types import (
    DataType,
    PreprocessingConfig,
    UserColumn,
    ArticleColumn,
)
import torch
import json
from utils.labelencoder import encode_labels
import numpy as np
from typing import Tuple
from config import only_users_and_articles_nodes


def save_to_csv(
    customers: pd.DataFrame, articles: pd.DataFrame, transactions: pd.DataFrame
):
    customers.to_csv("data/saved/customers.csv", index=False)
    articles.to_csv("data/saved/articles.csv", index=False)
    transactions.to_csv("data/saved/transactions.csv", index=False)


def preprocess(config: PreprocessingConfig):
    print("| Loading customers...")
    customers = pd.read_parquet("data/original/customers.parquet").fillna(0.0)
    customers = customers[[c.value for c in config.customer_features] + ["customer_id"]]

    print("| Loading articles...")
    articles = pd.read_parquet("data/original/articles.parquet").fillna(0.0)

    print("| Loading transactions...")
    transactions = pd.read_parquet("data/original/transactions_train.parquet")
    if config.data_size is not None:
        transactions = transactions[: config.data_size]

    print("| Calculating average price per product...")
    transactions_per_article = transactions.groupby(["article_id"]).mean()["price"]
    articles = articles.merge(
        transactions_per_article, on="article_id", how="outer"
    ).fillna(0.0)

    articles = articles[[c.value for c in config.article_features] + ["article_id"]]

    # There's currently a problem with k-core graph calculation: we'd need to re-map the edge indices after the nodes are removed and assumed a new ids.
    # if config.K > 0:
    #     print("| Adding transactions to the graph...")
    #     import networkit as nk

    #     node_features = pd.concat([customers, articles], axis=0)

    #     G = nk.Graph(n=node_features.shape[0])
    #     edge_pairs = zip(
    #         transactions["article_id"]
    #         .apply(lambda x: article_id_map_reverse[x])
    #         .to_numpy(),
    #         transactions["customer_id"]
    #         .apply(lambda x: customer_id_map_reverse[x])
    #         .to_numpy(),
    #     )
    #     for edge in tqdm(edge_pairs):
    #         G.addEdge(edge[0], edge[1])

    #     print("| Calculating the K-core of the graph...")
    #     original_node_count = len(node_features)
    #     k_core_per_node = sorted(nk.centrality.CoreDecomposition(G).run().ranking())
    #     nodes_to_remove = [row[0] for row in k_core_per_node if row[1] <= config.K]

    #     print("     Processing the about-to-be removed nodes...")
    #     # Remove the nodes from our records (node_features)
    #     node_features_to_remove = node_features.take(nodes_to_remove)
    #     node_features.drop(node_features.index[nodes_to_remove], axis=0, inplace=True)

    #     print("     Calculating the values for the to-be-removed edges...")
    #     # Remove the affected transactions (referring to missing nodes)
    #     customer_ids_to_remove = node_features_to_remove["customer_id"].unique()
    #     article_ids_to_remove = node_features_to_remove["article_id"].unique()

    #     print("     Remove the now redundant transactions...")
    #     transactions_to_remove_customers = transactions["customer_id"].isin(
    #         customer_ids_to_remove
    #     )
    #     transactions_to_remove_articles = transactions["article_id"].isin(
    #         article_ids_to_remove
    #     )
    #     transactions_to_remove = (
    #         transactions_to_remove_customers | transactions_to_remove_articles
    #     )
    #     transactions = transactions[~transactions_to_remove]

    #     print("     Remove the now redundant customers ...")
    #     customer_rows_to_remove = customers["customer_id"].isin(
    #         customer_ids_to_remove
    #     )
    #     customers = customers[~customer_rows_to_remove]

    #     print("     Remove the now redundant articles ...")
    #     article_rows_to_remove = articles["article_id"].isin(
    #         article_ids_to_remove
    #     )
    #     articles = articles[~article_rows_to_remove]

    #     print(
    #         f"     Number of nodes in the K-core: {len(node_features)}, kept: {round(len(node_features) / original_node_count, 2) * 100 }%"
    #     )

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

    print("| Removing disjoint nodes...")
    all_article_ids_referenced = set(transactions["article_id"].unique())
    all_customer_ids_referenced = set(transactions["customer_id"].unique())
    disjoint_customers = set(customers["customer_id"].unique()).difference(
        all_customer_ids_referenced
    )
    disjoint_articles = set(articles["article_id"].unique()).difference(
        all_article_ids_referenced
    )

    customers = customers[~customers["customer_id"].isin(disjoint_customers)]
    customers, customer_id_map_forward, customer_id_map_reverse = create_ids_and_maps(
        customers, "customer_id", 0
    )
    articles = articles[~articles["article_id"].isin(disjoint_articles)]
    articles, article_id_map_forward, article_id_map_reverse = create_ids_and_maps(
        articles,
        "article_id",
        0,
    )

    print("| Parsing transactions...")
    transactions_to_article_id = (
        transactions["article_id"].apply(lambda x: article_id_map_reverse[x]).to_numpy()
    )
    transactions_to_customer_id = (
        transactions["customer_id"]
        .apply(lambda x: customer_id_map_reverse[x])
        .to_numpy()
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
    data = create_func(
        customers,
        articles,
        transactions_to_customer_id,
        transactions_to_article_id,
    )

    print("| Saving the graph...")
    torch.save(data, "data/derived/graph.pt")
    if config.data_type == DataType.dgl:
        import dgl

        dgl.save_graphs("data/derived/graph.bin", [data], {"glabel": torch.tensor([0])})

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
    data["customer"].x = customers
    data["article"].x = articles
    data["customer", "buys", "article"].edge_index = torch.as_tensor(
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
            ("customer", "buys", "article"): (
                torch.as_tensor(transactions_to_customer_id, dtype=torch.long),
                torch.as_tensor(transactions_to_article_id, dtype=torch.long),
            )
        },
        num_nodes_dict={
            "customer": customers.shape[0],
            "article": articles.shape[0],
        },
    )
    data.nodes["customer"].data["features"] = customers
    data.nodes["article"].data["features"] = articles
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


if __name__ == "__main__":
    preprocess(only_users_and_articles_nodes)
