from neo4j import GraphDatabase
from typing import Optional

query_periodic_commit = "USING PERIODIC COMMIT 10000 "


class Database:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
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
        query = f"MATCH (n:{node_type}) " "WHERE id(n) = $node_id " "RETURN n as node"
        result = list(tx.run(query, node_id=node_id))
        return [row["node"] for row in result]

    def run_query(self, query: str):
        with self.driver.session() as session:
            info = session.run(query)
            print(info)

    def clear(self):
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)

    def create_indexes(self):
        self.run_query("CREATE INDEX ON :Customer(_id)")
        self.run_query("CREATE INDEX ON :Article(_id)")
