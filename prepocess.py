import pandas as pd
from tqdm import tqdm
from data.types import PreprocessingConfig, UserColumn, ArticleColumn
from igraph import *
from torch_geometric.data import Data
import torch


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

    node_features = customers[[c.value for c in config.customer_features]]

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

    articles, article_id_map_forward, article_id_map_reverse = create_ids_and_maps(
        articles, "article_id", len(customer_id_map_forward)
    )

    # TODO: add back the ability to create nodes from node features
    # for column in config.article_nodes:
    #     G.add_nodes_from(articles[column.value])
    #     G.add_edges_from(zip(articles["article_id"], articles[column.value]))

    article_features = articles[[c.value for c in config.article_features]]
    node_features = pd.concat([node_features, article_features], axis=0)

    print("| Adding transactions to the graph...")
    G = Graph(n=node_features.shape[0], vertex_attrs=node_features.to_dict())

    edge_pairs = zip(
        transactions["article_id"].apply(lambda x: article_id_map_reverse[x]),
        transactions["customer_id"].apply(lambda x: customer_id_map_reverse[x]),
    )
    G.add_edges(edge_pairs)

    print("| Calculating the K-core of the graph...")
    original_node_count = G.vcount()
    G = G.k_core(config.K)
    print(
        f"     Number of nodes in the K-core: {G.vcount()}, kept: {round(G.vcount() / original_node_count, 2)}%"
    )

    print("| Converting the graph to torch-geometric format...")
    adj_matrix = G.get_adjacency_sparse()
    vertex_features = G.vs.attributes()
    data = Data(edge_index=adj_matrix)

    print("| Saving the graph...")
    torch.save(data, "data/derived/graph.pt")


# TODO: remove this when format stabilizes
def create_prefixed_values_df(df: pd.DataFrame, prefix_mapping: dict):
    for key, value in tqdm(prefix_mapping.items()):
        df[key] = df[key].apply(lambda x: value + str(x))
    return df


def create_ids_and_maps(
    df: pd.DataFrame, column: str, start: int
) -> tuple[pd.DataFrame, dict, dict]:
    df.index += start
    mapping_forward = df[column].to_dict()
    mapping_reverse = {v: k for k, v in mapping_forward.items()}
    df.drop(column, axis=1, inplace=True)
    df.reset_index(inplace=True)
    return df, mapping_forward, mapping_reverse


only_users_and_articles_nodes = PreprocessingConfig(
    customer_features=[
        UserColumn.PostalCode,
        # UserColumn.FN,
        # UserColumn.Age,
        # UserColumn.ClubMemberStatus,
        # UserColumn.FashionNewsFrequency,
        # UserColumn.Active,
    ],
    # customer_nodes=[],
    article_features=[
        ArticleColumn.ProductCode,
        # ArticleColumn.ProductTypeNo,
        # ArticleColumn.GraphicalAppearanceNo,
        # ArticleColumn.ColourGroupCode,
    ],
    # article_nodes=[],
    K=20,
    data_size=10000000,
)

preprocess(only_users_and_articles_nodes)
