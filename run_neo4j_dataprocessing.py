from data.graph_database.neo4j_advanced import App
import torch as t
import pandas as pd


db = App(uri="bolt://localhost:7687", user="neo4j", password="123456")


unified_transactions = pd.read_parquet("data/original/transactions_splitted.parquet")[
    :100000
]
db.create_constraints()
# db.create_entire_database(unified_transactions)
db.create_entire_database_efficient(unified_transactions)

# db.find_node("User 3", "User")


db.close()
