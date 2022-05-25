from neo4j import GraphDatabase
from typing import Optional

query_periodic_commit = "USING PERIODIC COMMIT 10000 "


class Database:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    """ GET """

    @staticmethod
    def query_node(node_id: int, node_type: str, no_return: bool = False) -> str:
        query = f"MATCH(n:{node_type} {{_id:'{str(node_id)}'}})"

        if no_return:
            return query + " "
        else:
            return query + " RETURN n"

    @staticmethod
    def query_n_neighbors(
        node_id: int,
        n_neighbor: int,
        node_type: str,
        split_type: str,
        start_neighbor: int = 0,
        no_return: bool = False,
    ) -> str:
        split_string = split_type + "_mask"
        # slow_query = f"MATCH(n:{node_type} {{_id:'{str(node_id)}'}}) WITH n MATCH(n)-[r:BUYS*1..{str(n_neighbor)}{{{split_string}:'1'}}]-(m)"
        # query_traditional = f"MATCH(n:{node_type} {{_id:'{str(node_id)}'}}) WITH n MATCH(n)-[r:BUYS*1..{str(n_neighbor)}{{{split_string}:'1'}}]-(m) UNWIND r AS rel WITH DISTINCT rel"
        # query = (
        #     f"MATCH (p:Customer {{_id: '{str(node_id)}'}})"
        #     + f" CALL apoc.path.subgraphAll(p, {{relationshipFilter: '<BUYS {{{split_string}:'1'}}', minLevel: 1, maxLevel: {str(n_neighbor)}}})"
        #     + " YIELD relationships"
        # )

        base = "BUYS_TRAIN"
        extension = (
            "|BUYS_VAL"
            if split_type == "val"
            else "|BUYS_VAL|BUYS_TEST"
            if split_type == "test"
            else ""
        )
        extra = "|HAS_COLOR"
        rel_string = base + extension + extra

        query = (
            f"MATCH (p:Customer {{_id: '{str(node_id)}'}}) "
            + f" CALL apoc.path.subgraphAll(p, {{relationshipFilter: '{rel_string}', minLevel: {str(start_neighbor)}, maxLevel: {str(n_neighbor)}}})"
            + f" YIELD relationships"
            + f" RETURN [r in relationships | [LABELS(STARTNODE(r))[0],TYPE(r),LABELS(ENDNODE(r))[0], STARTNODE(r)._id,ENDNODE(r)._id]] as arraysomething"
        )

        if no_return:
            return query + " "
        else:
            return query + " RETURN relationships"

    @staticmethod
    def query_all_nodes(node_type: str) -> str:
        query = f"MATCH (n:{node_type}) RETURN n"

        return query

    """ UTILITY METHODS """

    def run_match(self, query: str):
        with self.driver.session() as session:
            result = list(session.run(query))

            return result

    def run_query(self, query: str):
        with self.driver.session() as session:
            info = session.run(query)
            print(info)
            return list(info)

    def clear(self):
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)

    def create_indexes(self):
        self.run_query("CREATE INDEX ON :Customer(_id)")
        self.run_query("CREATE INDEX ON :Article(_id)")
