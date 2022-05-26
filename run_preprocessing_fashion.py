import pandas as pd
from tqdm import tqdm
from data.types import (
    DataType,
    PreprocessingConfig,
)
import torch as t
import json
from utils.labelencoder import encode_labels
from config import preprocessing_config
from utils.preprocessing import (
    create_data_pyg,
    create_data_dgl,
    create_ids_and_maps,
    extract_edges,
    extract_reverse_edges,
)
from utils.constants import Constants
from data.neo4j.save import save_to_neo4j


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

    extra_nodes = None
    extra_edges = None
    if Constants.node_extra is not None:
        print("| Loading extra node type...")
        extra_nodes = pd.DataFrame(
            articles[Constants.node_extra].unique(),
            columns=[Constants.node_extra],
        )
        (
            extra_nodes,
            extra_nodes_id_map_forward,
            extra_nodes_id_map_reverse,
        ) = create_ids_and_maps(
            extra_nodes,
            Constants.node_extra,
            0,
        )
        extra_edges = articles[["article_id", Constants.node_extra]]
        extra_edges[Constants.node_extra] = extra_edges[Constants.node_extra].apply(
            lambda x: extra_nodes_id_map_reverse[x]
        )
        extra_edges["article_id"] = extra_edges["article_id"].apply(
            lambda x: article_id_map_reverse[x]
        )
        extra_edges.rename(
            columns={Constants.node_extra: f"{Constants.node_extra}_id"}, inplace=True
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
    if extra_nodes is not None:
        extra_nodes.drop([Constants.node_extra], axis=1, inplace=True)

    if config.save_to_neo4j:
        save_to_neo4j(
            customers,
            articles,
            transactions,
            extra_nodes,
            Constants.node_extra if Constants.node_extra is not None else None,
            extra_edges if extra_edges is not None else None,
            Constants.rel_type_extra,
        )

    print("| Converting to tensors...")
    customers = t.tensor(customers.to_numpy(), dtype=t.long)
    assert t.isnan(customers).any() == False

    articles = t.tensor(articles.to_numpy(), dtype=t.long)
    if config.load_image_embedding:
        articles = t.cat((articles, per_article_img_embedding), axis=1)
    if config.load_text_embedding:
        articles = t.cat((articles, per_article_text_embedding), axis=1)
    assert t.isnan(articles).any() == False

    if Constants.node_extra is not None:
        extra_nodes = t.tensor(extra_nodes.to_numpy(), dtype=t.long)

    print("| Creating Data...")
    # If we ever want to get dgl data creation back
    # create_func = (
    #     create_data_dgl if config.data_type == DataType.dgl else create_data_pyg
    # )
    train_graph = create_data_pyg(
        customers,
        articles,
        extra_nodes,
        Constants.node_extra if Constants.node_extra is not None else None,
        transactions_train["customer_id"].to_numpy(),
        transactions_train["article_id"].to_numpy(),
        extra_edges["article_id"].to_numpy() if extra_edges is not None else None,
        extra_edges[f"{Constants.node_extra}_id"].to_numpy()
        if extra_edges is not None
        else None,
        Constants.edge_key_extra,
    )
    val_graph = create_data_pyg(
        customers,
        articles,
        extra_nodes,
        Constants.node_extra if Constants.node_extra is not None else None,
        transactions_val["customer_id"].to_numpy(),
        transactions_val["article_id"].to_numpy(),
        extra_edges["article_id"].to_numpy() if extra_edges is not None else None,
        extra_edges[f"{Constants.node_extra}_id"].to_numpy()
        if extra_edges is not None
        else None,
        Constants.edge_key_extra,
    )
    test_graph = create_data_pyg(
        customers,
        articles,
        extra_nodes,
        Constants.node_extra if Constants.node_extra is not None else None,
        transactions_test["customer_id"].to_numpy(),
        transactions_test["article_id"].to_numpy(),
        extra_edges["article_id"].to_numpy() if extra_edges is not None else None,
        extra_edges[f"{Constants.node_extra}_id"].to_numpy()
        if extra_edges is not None
        else None,
        Constants.edge_key_extra,
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


if __name__ == "__main__":
    preprocess(preprocessing_config)
