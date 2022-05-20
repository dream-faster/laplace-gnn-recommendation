import pandas as pd
import os
import time
from utils.pandas import drop_columns_if_exist
import numpy as np
from typing import Optional, Tuple
import torch as t


def save_to_csv(dataframe: pd.DataFrame, name: str):
    dataframe.to_csv(f"data/saved/{name}.csv", index=False)


def save_to_neo4j(
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    transactions: pd.DataFrame,
    extra_nodes: Optional[pd.DataFrame],
    extra_node_name: Optional[str],
    extra_edges: Optional[pd.DataFrame],
    extra_edge_type_label: Optional[str],
):
    print("| Saving to neo4j...")
    print("| Processing customer nodes...")
    customers = customers.copy()
    customers[":LABEL"] = "Customer"
    customers.rename(columns={"index": ":ID(Customer)"}, inplace=True)
    customers["_id"] = customers[":ID(Customer)"]
    save_to_csv(customers, "customers")

    print("| Processing article nodes...")
    articles = articles.copy()
    articles[":LABEL"] = "Article"
    articles.rename(columns={"index": ":ID(Article)"}, inplace=True)
    articles["_id"] = articles[":ID(Article)"]
    save_to_csv(articles, "articles")

    print("| Renaming transactions...")
    transactions = transactions.copy()
    transactions.rename(
        columns={
            "customer_id": ":START_ID(Customer)",
            "article_id": ":END_ID(Article)",
        },
        inplace=True,
    )

    if extra_nodes is not None:
        print("| Processing extra nodes...")
        extra_node_df = pd.DataFrame(data=extra_nodes.copy())
        extra_node_df[":LABEL"] = extra_edge_type_label
        extra_node_df.rename(columns={"index": f":ID({extra_node_name})"}, inplace=True)
        extra_node_df["_id"] = extra_node_df[f":ID({extra_node_name})"]
        extra_node_df = extra_node_df.astype(str)
        save_to_csv(extra_node_df, f"{extra_node_name}")
        new_extra_edges = extra_edges.copy()
        new_extra_edges.rename(
            columns={
                "article_id": ":START_ID(Article)",
                "extra_node_id": f":END_ID({extra_node_name})",
            },
            inplace=True,
        )
        new_extra_edges = new_extra_edges.astype(int)
        new_extra_edges[":TYPE"] = extra_edge_type_label
        save_to_csv(new_extra_edges, "extra_transactions")

    transactions = drop_columns_if_exist(
        transactions, ["t_dat", "price", "sales_channel_id", "year-month"]
    )

    transactions["train_mask"] = transactions["train_mask"].astype(int)
    transactions["test_mask"] = transactions["test_mask"].astype(int)
    transactions["val_mask"] = transactions["val_mask"].astype(int)

    print("| Changing the edge names...")
    transactions[":TYPE"] = transactions.apply(
        lambda x: "BUYS_TEST"
        if x["test_mask"] == 1
        else "BUYS_VAL"
        if x["val_mask"] == 1
        else "BUYS_TRAIN",
        axis=1,
    )

    save_to_csv(transactions, "transactions")
    # Neo4j needs to be stopped for neo4j-admin import to run
    print("| Stopping running instances of Neo4j...")
    os.system("neo4j stop")
    print("| Importing csv to database...")
    if extra_nodes is not None:
        os.system(
            f"neo4j-admin import --database=neo4j --nodes=data/saved/articles.csv --nodes=data/saved/customers.csv --nodes=data/saved/{extra_node_name}.csv --relationships=data/saved/transactions.csv --relationships=data/saved/extra_transactions.csv --force"
        )
    else:
        os.system(
            f"neo4j-admin import --database=neo4j --nodes=data/saved/articles.csv --nodes=data/saved/customers.csv --relationships=data/saved/transactions.csv --force"
        )

    print("| Starting Neo4j...")
    os.system("neo4j start")
    time.sleep(10)
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
    )
    os.system(
        "echo 'CREATE FULLTEXT INDEX relationship_index FOR ()-[r:BUYS]-() ON EACH [r.train_mask]' | cypher-shell -u neo4j -p password --format plain"
    )

    print("Number of nodes in the database:")
    os.system(
        "echo 'MATCH (n) RETURN count(n)' | cypher-shell -u neo4j -p password --format plain"
    )
