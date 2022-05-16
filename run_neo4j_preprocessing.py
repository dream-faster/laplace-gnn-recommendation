from data.graph_database.neo4j_advanced import App
import torch as t
import pandas as pd

""" Preprocessing steps for command line """
unified_transactions = pd.read_parquet("data/original/transactions_splitted.parquet")

articles = pd.DataFrame(unified_transactions["article_id"].unique())
customers = pd.DataFrame(unified_transactions["customer_id"].unique())

unified_transactions.to_csv("data/original/transactions_splitted.csv", index=False)
articles.to_csv("data/original/articles.csv", index=False)
customers.to_csv("data/original/customers.csv", index=False)

""" Through python api """
# db = App(uri="bolt://localhost:7687", user="neo4j", password="123456")
# db.create_constraints()
# db.create_entire_database(unified_transactions)
# db.create_entire_database_batched(unified_transactions)
# db.create_entire_database_batched(None)

# db.close()
