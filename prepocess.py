import pandas as pd
from tqdm import tqdm
from data.types import PreprocessingConfig, UserColumn, ArticleColumn
from torch_geometric.utils.convert import from_networkx
import torch
import networkit as nk
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import json
from utils.labelencoder import encode_labels


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
    # TODO: if we want to get k-core working, we need to use networkit (but there are some issues there)
    # G = nk.Graph(n=node_features.shape[0])
    # edge_pairs = zip(
    #     transactions["article_id"].apply(lambda x: article_id_map_reverse[x]),
    #     transactions["customer_id"].apply(lambda x: customer_id_map_reverse[x]),
    # )
    # for edge in tqdm(edge_pairs):
    #     G.addEdge(edge[0], edge[1])

    # print("| Calculating the K-core of the graph...")
    # original_node_count = G.numberOfNodes()
    # k_core_per_node = sorted(nk.centrality.CoreDecomposition(G).run().ranking())
    # k_core_per_node = [row[0] for row in k_core_per_node if row[1] >= config.K]
    # for node in tqdm(k_core_per_node):
    #     G.removeNode(node)
    # node_features.drop(node_features.index[k_core_per_node], axis=0, inplace=True)

    # print(
    #     f"     Number of nodes in the K-core: {G.numberOfNodes()}, kept: {round(G.numberOfNodes() / original_node_count, 2) * 100 }%"
    # )

    # print("| Converting the graph to torch-geometric format...")
    # G = nk.nxadapter.nk2nx(G)

    print("| Encoding features...")
    for column in tqdm(node_features.columns):
        node_features[column] = encode_labels(node_features[column])
        
    node_features = node_features.reset_index().to_numpy()
    node_features = torch.tensor(node_features, dtype=torch.long)

    print("| Creating PyG Data...")
    data = Data(
        x=node_features,
        edge_index=torch.Tensor(
            [
                transactions["article_id"].apply(lambda x: article_id_map_reverse[x])
                + transactions["customer_id"].apply(
                    lambda x: customer_id_map_reverse[x]
                ),
                transactions["customer_id"].apply(lambda x: customer_id_map_reverse[x])
                + transactions["article_id"].apply(lambda x: article_id_map_reverse[x]),
            ]
        ),
    )

    print("| Saving the graph...")
    torch.save(data, "data/derived/graph.pt")

    print("| Saving the node-to-id mapping...")
    with open("data/derived/customer_id_map_forward.csv", "w") as fp:
        json.dump(customer_id_map_forward, fp)
    with open("data/derived/article_id_map_forward.csv", "w") as fp:
        json.dump(article_id_map_forward, fp)


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
    K=0,
    data_size=None,
)

preprocess(only_users_and_articles_nodes)
