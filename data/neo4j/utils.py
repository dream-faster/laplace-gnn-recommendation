from neo4j.graph import Node, Relationship
from data.neo4j.neo4j_database import Database
from utils.constants import Constants
from collections import defaultdict


def get_neighborhood(
    db: Database, node_id: int, n_neighbor: int, split_type: str
) -> list[list]:
    result = db.run_match(
        db.query_n_neighbors(
            node_id=node_id,
            n_neighbor=n_neighbor,
            node_type="Customer",
            split_type=split_type,
            no_return=True,
        )
    )

    """ Filter nodes and experiments into lists """
    edge_index = defaultdict(list)

    for from_type, rel_type, to_type, from_id, to_id in result[0][0]:
        edge_index[(from_type, rel_type, to_type)].append((from_id, to_id))

    for key, index in edge_index.items():
        edge_index[key] = list(map(list, zip(*index)))

    return edge_index


def get_id_map(db: Database) -> tuple[dict, dict]:
    customers = db.run_match(db.query_all_nodes(node_type="Customer"))
    articles = db.run_match(db.query_all_nodes(node_type="Article"))

    customer_map = {customer["n"].id: customer["n"]._id for customer in customers}
    article_map = {article["n"].id: article["n"]._id for article in articles}

    return customer_map, article_map
