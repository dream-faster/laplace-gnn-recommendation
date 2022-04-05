import pandas as pd
from tqdm import tqdm
from data.types import PreprocessingConfig, UserColumn, ArticleColumn
import torch
from torch_geometric.data import Data
import json
from utils.labelencoder import encode_labels
import numpy as np
from typing import Tuple


def preprocess(config: PreprocessingConfig):
    print("| Loading customers...")
    customers = pd.read_parquet("data/original/customers.parquet").fillna(0.0)
    print("| Transforming customers...")
    customers, customer_id_map_forward, customer_id_map_reverse = create_ids_and_maps(
        customers, "customer_id", 0
    )

    print("| Adding customer features...")
    # TODO: add back the ability to create nodes from node features
    # for column in config.customer_nodes:
    #     G.add_nodes_from(customers[column.value])
    #     G.add_edges_from(zip(customers["index"], customers[column.value]))

    node_features = customers[
        [c.value for c in config.customer_features] + ["customer_id"]
    ]

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

    print("| Loading article image embeddings...")
    articles_image_embeddings = torch.load(
        "data/derived/fashion-recommendation-image-embeddings-clip-ViT-B-32.pt"
    )
    articles["img_embedding"] = articles.apply(
        lambda article: articles_image_embeddings.get(
            int(article["article_id"]), torch.zeros(512)
        ),
        axis=1,
    )

    print("| Loading article text embeddings...")
    articles_text_embeddings = torch.load(
        "data/derived/fashion-recommendation-text-embeddings-clip-ViT-B-32.pt"
    )
    for key in articles_text_embeddings[108775015].keys():
        articles[key] = articles.apply(
            lambda article: articles_text_embeddings[int(article["article_id"])].get(
                key, torch.zeros(512)
            ),
            axis=1,
        )

    articles, article_id_map_forward, article_id_map_reverse = create_ids_and_maps(
        articles, "article_id", len(customer_id_map_forward)
    )

    # TODO: add back the ability to create nodes from node features
    # for column in config.article_nodes:
    #     G.add_nodes_from(articles[column.value])
    #     G.add_edges_from(zip(articles["article_id"], articles[column.value]))

    article_features = articles[
        [c.value for c in config.article_features] + ["article_id"]
    ]
    node_features = pd.concat([node_features, article_features], axis=0)

    if config.K > 0:
        print("| Adding transactions to the graph...")
        import networkit as nk

        G = nk.Graph(n=node_features.shape[0])
        edge_pairs = zip(
            transactions["article_id"]
            .apply(lambda x: article_id_map_reverse[x])
            .to_numpy(),
            transactions["customer_id"]
            .apply(lambda x: customer_id_map_reverse[x])
            .to_numpy(),
        )
        for edge in tqdm(edge_pairs):
            G.addEdge(edge[0], edge[1])

        print("| Calculating the K-core of the graph...")
        original_node_count = len(node_features)
        k_core_per_node = sorted(nk.centrality.CoreDecomposition(G).run().ranking())
        nodes_to_remove = [row[0] for row in k_core_per_node if row[1] <= config.K]

        print("     Processing the about-to-be removed nodes...")
        # Remove the nodes from our records (node_features)
        node_features_to_remove = node_features.take(nodes_to_remove)
        node_features.drop(node_features.index[nodes_to_remove], axis=0, inplace=True)

        print("     Calculating the values for the to-be-removed edges...")
        # Remove the affected transactions (referring to missing nodes)
        customer_ids_to_remove = node_features_to_remove["customer_id"].unique()
        article_ids_to_remove = node_features_to_remove["article_id"].unique()

        print("     Get the indicies of the transactions to be removed...")
        transactions_to_remove_customers = transactions["customer_id"].isin(
            customer_ids_to_remove
        )
        transactions_to_remove_articles = transactions["article_id"].isin(
            article_ids_to_remove
        )
        transactions_to_remove = (
            transactions_to_remove_customers | transactions_to_remove_articles
        )
        transactions = transactions[~transactions_to_remove]
        print(
            f"     Number of nodes in the K-core: {len(node_features)}, kept: {round(len(node_features) / original_node_count, 2) * 100 }%"
        )

    print("     Remove unused columns...")
    node_features.drop(["customer_id", "article_id"], axis=1, inplace=True)

    print("| Encoding features...")
    for column in tqdm(node_features.columns):
        if column not in config.article_non_categorical_features:
            node_features[column] = encode_labels(node_features[column])

    node_features = node_features.reset_index().to_numpy()
    node_features = torch.tensor(node_features, dtype=torch.long)

    transactions_to_article_id = (
        transactions["article_id"].apply(lambda x: article_id_map_reverse[x]).to_numpy()
    )
    transactions_to_customer_id = (
        transactions["customer_id"]
        .apply(lambda x: customer_id_map_reverse[x])
        .to_numpy()
    )

    print("| Creating PyG Data...")
    data = Data(
        x=node_features,
        edge_index=torch.as_tensor(
            np.array(
                [
                    np.concatenate(
                        (transactions_to_article_id, transactions_to_customer_id)
                    ),
                    np.concatenate(
                        (transactions_to_customer_id, transactions_to_article_id)
                    ),
                ]
            ),
            dtype=torch.long,
        ),
    )

    print("| Saving the graph...")
    torch.save(data, "data/derived/graph.pt")

    print("| Saving the node-to-id mapping...")
    with open("data/derived/customer_id_map_forward.json", "w") as fp:
        json.dump(customer_id_map_forward, fp)
    with open("data/derived/article_id_map_forward.json", "w") as fp:
        json.dump(article_id_map_forward, fp)


# TODO: remove this when format stabilizes
def create_prefixed_values_df(df: pd.DataFrame, prefix_mapping: dict):
    for key, value in tqdm(prefix_mapping.items()):
        df[key] = df[key].apply(lambda x: value + str(x))
    return df


def create_ids_and_maps(
    df: pd.DataFrame, column: str, start: int
) -> Tuple[pd.DataFrame, dict, dict]:
    df.index += start
    mapping_forward = df[column].to_dict()
    mapping_reverse = {v: k for k, v in mapping_forward.items()}
    df["index"] = df.index
    return df, mapping_forward, mapping_reverse


only_users_and_articles_nodes = PreprocessingConfig(
    customer_features=[
        UserColumn.PostalCode,
        UserColumn.FN,
        UserColumn.Age,
        UserColumn.ClubMemberStatus,
        UserColumn.FashionNewsFrequency,
        UserColumn.Active,
    ],
    # customer_nodes=[],
    article_features=[
        ArticleColumn.ProductCode,
        ArticleColumn.ProductTypeNo,
        ArticleColumn.GraphicalAppearanceNo,
        ArticleColumn.ColourGroupCode,
    ],
    # article_nodes=[],
    article_non_categorical_features=[ArticleColumn.ImgEmbedding],
    K=0,
    data_size=None,
)

if __name__ == "__main__":
    preprocess(only_users_and_articles_nodes)
