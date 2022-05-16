import logging
import sys

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.enable_log(logging.INFO, sys.stdout)

    def close(self):
        # Don't forget to close the driver connection when you are finished with it
        self.driver.close()

    @staticmethod
    def enable_log(level, output_stream):
        handler = logging.StreamHandler(output_stream)
        handler.setLevel(level)
        logging.getLogger("neo4j").addHandler(handler)
        logging.getLogger("neo4j").setLevel(level)

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

    def create_constraints(self):
        with self.driver.session() as session:
            # Write transactions allow the driver to handle retries and transient errors
            session.write_transaction(self._create_constraints)

    @staticmethod
    def _create_constraints(tx):
        tx.run(
            "CREATE CONSTRAINT unique_user_id IF NOT EXISTS FOR (user:User) REQUIRE user._id IS UNIQUE"
        )
        tx.run(
            "CREATE CONSTRAINT unique_article_id IF NOT EXISTS FOR (article:Article) REQUIRE article._id IS UNIQUE"
        )

    @staticmethod
    def _create_and_return_transaction(tx, user_id, article_id):
        # To learn more about the Cypher syntax, see https://neo4j.com/docs/cypher-manual/current/
        # The Reference Card is also a good resource for keywords https://neo4j.com/docs/cypher-refcard/current/

        query = (
            "MERGE (p:User { _id: $user_id }) "
            "MERGE (a:Article { _id: $article_id }) "
            "MERGE (p)-[k:BUYS]->(a) "
            "RETURN p, a"
        )
        result = tx.run(
            query,
            user_id=user_id,
            article_id=article_id,
        )
        try:
            return [
                {
                    "p": row["p"]["_id"],
                    "a": row["a"]["_id"],
                }
                for row in result
            ]
        # Capture any errors along with the query and data for traceability
        except ServiceUnavailable as exception:
            logging.error(
                "{query} raised an error: \n {exception}".format(
                    query=query, exception=exception
                )
            )
            raise

    def create_transaction(self, user_name, article_name):
        with self.driver.session() as session:
            # Write transactions allow the driver to handle retries and transient errors
            result = session.write_transaction(
                self._create_and_return_transaction,
                user_name,
                article_name,
            )

    def create_entire_database(self, transactions_parquet: pd.DataFrame):
        with self.driver.session().begin_transaction() as tx:
            transactions_parquet.progress_apply(
                lambda x: self._create_and_return_transaction(
                    tx, x["customer_id"], x["article_id"]
                ),
                axis=1,
            )
            tx.commit()


if __name__ == "__main__":
    app = App(uri="bolt://localhost:7687", user="neo4j", password="123456")

    app.create_constraints()
    app.create_transaction("User 3", "Article 2")
    app.find_node("User 3", "User")
    app.close()
