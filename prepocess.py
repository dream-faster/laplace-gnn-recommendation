from dataclasses import dataclass
from enum import Enum
import pandas as pd
import networkx as nx



class UserColumn(Enum):
    PostalCode = "postal_code"
    FN = "FN"
    Age = "age"
    ClubMemberStatus = "club_member_status"
    FashionNewsFrequency = "fashion_news_frequency"
    Active = "active"

class ArticleColumn(Enum):
    ProductCode = "product_code"
    ProductTypeNo = "product_type_no"
    GraphicalAppearanceNo = "graphical_appearance_no"
    ColourGroupCode = "colour_group_code"


@dataclass
class PreprocessingConfig():
    customer_features: list[UserColumn]
    customer_nodes: list[UserColumn]

    article_features: list[ArticleColumn]
    article_nodes: list[ArticleColumn]



def preprocess(config: PreprocessingConfig):
    customers = pd.read_parquet('data/customers.parquet').fillna(0.)
    G_customers = nx.Graph()
    G_customers.add_nodes_from(["C-" + c for c in customers['customer_id']])
    G_customers.add_nodes_from(["Post-" + p for p in customers['postal_code']])
    G_customers.add_edges_from([("C-" + c, "Post-" + p) for c, p in zip(customers['customer_id'], customers['postal_code'])])

    articles = pd.read_parquet('data/articles.parquet').fillna(0.)
    transactions = pd.read_parquet('data/transactions_train.parquet')

    transactions_per_article = transactions.groupby(['article_id']).mean()['price'].to_dict()

    G_articles = nx.Graph()
    G_articles.add_nodes_from(["A-" + str(c) for c in articles['article_id']])
    G_articles.add_nodes_from(["Product-" + str(p) for p in articles['product_code']])
    G_articles.add_edges_from([("A-" + str(a), "Product-" + str(p)) for a, p in zip(articles['article_id'], articles['product_code'])])

    G = G_customers.update(G_articles.nodes, G_articles.edges)
    print(G.nodes)

preprocess(PreprocessingConfig(customer_features=[UserColumn.Active], customer_nodes=[UserColumn.PostalCode, UserColumn.Age], article_features=[], article_nodes=[]))