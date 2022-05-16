from neo4j import GraphDatabase
from typing import Optional

query_periodic_commit = "USING PERIODIC COMMIT 10000 "


class App:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    """ GET """

    def find_node(self, node_id, node_type):
        with self.driver.session() as session:
            result = session.read_transaction(
                self._find_and_return_node, node_id, node_type
            )
            for row in result:
                print("Found user: {row}".format(row=row))

    @staticmethod
    def _find_and_return_node(tx, node_id, node_type):
        query = f"MATCH (n:{node_type}) " "WHERE n._id = $node_id " "RETURN n as node"
        result = list(tx.run(query, node_id=node_id))
        return [row["node"] for row in result]

    """ CREATE """

    def run_query(self, query: str):
        with self.driver.session() as session:
            info = session.run(query)
            print(info)

    def create_constraints(self):
        self.run_query(
            "CREATE CONSTRAINT unique_user_id IF NOT EXISTS FOR (user:User) REQUIRE user._id IS UNIQUE"
        )
        self.run_query(
            "CREATE CONSTRAINT unique_article_id IF NOT EXISTS FOR (article:Article) REQUIRE article._id IS UNIQUE"
        )

    def load_articles_csv(self):
        query = (
            query_periodic_commit
            + "LOAD CSV FROM 'https://storage.googleapis.com/heii-public/neo4j/articles.csv' AS row WITH row[0] AS id MERGE (a:Article {_id: id}) RETURN count(a);"
        )
        self.run_query(query)

    def load_customers_csv(self):
        query = (
            query_periodic_commit
            + "LOAD CSV FROM 'https://storage.googleapis.com/heii-public/neo4j/customers.csv' AS row WITH row[0] AS id MERGE (c:Customer {_id: id}) RETURN count(c);"
        )
        self.run_query(query)

    def load_relationships(self):
        query = (
            query_periodic_commit
            + "LOAD CSV WITH HEADERS FROM 'https://storage.googleapis.com/heii-public/neo4j/transactions.csv' AS row WITH row.customer_id AS customer_id, row.article_id AS article_id MATCH (c:Customer {_id: customer_id}) MATCH (a:Article {_id: article_id}) MERGE (c)-[rel:BUYS]->(a) RETURN count(rel);"
        )
        self.run_query(query)

    def clear(self):
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)

    def create_indexes(self):
        self.run_query("CREATE INDEX ON :Customer(_id)")
        self.run_query("CREATE INDEX ON :Article(_id)")
