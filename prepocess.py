from dataclasses import dataclass
from enum import Enum
import pandas as pd
import networkx as nx
from tqdm import tqdm


class UserColumn(Enum):
    PostalCode = "postal_code"
    FN = "FN"
    Age = "age"
    ClubMemberStatus = "club_member_status"
    FashionNewsFrequency = "fashion_news_frequency"
    Active = "Active"


class ArticleColumn(Enum):
    ProductCode = "product_code"
    ProductTypeNo = "product_type_no"
    GraphicalAppearanceNo = "graphical_appearance_no"
    ColourGroupCode = "colour_group_code"
    AvgPrice = "avg_price"


@dataclass
class PreprocessingConfig:
    customer_features: list[UserColumn]
    customer_nodes: list[UserColumn]

    article_features: list[ArticleColumn]
    article_nodes: list[ArticleColumn]

    K: int


def preprocess(config: PreprocessingConfig):
    print("| Loading customers...")
    customers = pd.read_parquet("data/customers.parquet").fillna(0.0)
    print("| Transforming customers...")
    customers = create_prefixed_values_df(
        customers,
        {
            "customer_id": "C-",
            UserColumn.PostalCode.value: "Post-",
            UserColumn.FN.value: "FN-",
            UserColumn.Age.value: "Age-",
            UserColumn.ClubMemberStatus.value: "Club-",
            UserColumn.FashionNewsFrequency.value: "News-",
            UserColumn.Active.value: "Active-",
        },
    )

    print("| Adding customers to the graph...")
    G = nx.Graph()
    G.add_nodes_from(customers["customer_id"])

    for column in config.customer_nodes:
        G.add_nodes_from(customers[column.value])
        G.add_edges_from(zip(customers["customer_id"], customers[column.value]))

    for column in config.customer_features:
        nx.set_node_attributes(G, customers[column.value].to_dict(), column.value)

    print("| Loading articles...")
    articles = pd.read_parquet("data/articles.parquet").fillna(0.0)

    print("| Loading transactions...")
    transactions = pd.read_parquet("data/transactions_train.parquet")
    # transactions = transactions[:10000]
    print("| Transforming transactions...")
    transactions = create_prefixed_values_df(
        transactions,
        {
            "article_id": "A-",
            "customer_id": "C-",
        },
    )

    print("| Calculating average price per product...")
    transactions_per_article = (
        transactions.groupby(["article_id"]).mean()["price"].to_dict()
    )
    articles["avg_price"] = 0.0
    for article_id, price in tqdm(transactions_per_article.items()):
        articles.loc[articles["article_id"] == article_id, "avg_price"] = price

    print("| Transforming articles...")
    articles = create_prefixed_values_df(
        articles,
        {
            "article_id": "A-",
            ArticleColumn.ProductCode.value: "PCode-",
            ArticleColumn.ProductTypeNo.value: "PType-",
            ArticleColumn.GraphicalAppearanceNo.value: "Appea-",
            ArticleColumn.ColourGroupCode.value: "Colour-",
            ArticleColumn.AvgPrice.value: "Price-",
        },
    )

    print("| Adding articles to the graph...")
    G.add_nodes_from(articles["article_id"])

    for column in config.article_nodes:
        G.add_nodes_from(articles[column.value])
        G.add_edges_from(zip(articles["article_id"], articles[column.value]))

    for column in config.article_features:
        nx.set_node_attributes(G, articles[column.value].to_dict(), column.value)

    print("| Adding transactions to the graph...")
    G.add_edges_from(zip(transactions["article_id"], transactions["customer_id"]))

    print("| Calculating the K-core of the graph...")
    G = nx.k_core(G, config.K)

    print("| Saving the graph...")
    nx.write_gpickle(G, "data/graph.gpickle")


def create_prefixed_values_df(df: pd.DataFrame, prefix_mapping: dict):
    for key, value in tqdm(prefix_mapping.items()):
        df[key] = df[key].apply(lambda x: value + str(x))
    return df


only_users_and_articles_nodes = PreprocessingConfig(
    customer_features=[
        UserColumn.PostalCode,
        # UserColumn.FN,
        # UserColumn.Age,
        # UserColumn.ClubMemberStatus,
        # UserColumn.FashionNewsFrequency,
        # UserColumn.Active,
    ],
    customer_nodes=[],
    article_features=[
        ArticleColumn.ProductCode,
        # ArticleColumn.ProductTypeNo,
        # ArticleColumn.GraphicalAppearanceNo,
        # ArticleColumn.ColourGroupCode,
    ],
    article_nodes=[],
    K=10,
)

preprocess(only_users_and_articles_nodes)
