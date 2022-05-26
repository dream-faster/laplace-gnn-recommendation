from neo4j.graph import Node, Relationship
from data.neo4j.neo4j_database import Database
from utils.constants import Constants
from collections import defaultdict
import torch as t


def get_neighborhood(
    db: Database, node_id: int, n_neighbor: int, start_neighbor: int, split_type: str
) -> defaultdict:
    result = db.run_match(
        db.query_n_neighbors(
            node_id=node_id,
            n_neighbor=n_neighbor,
            node_type="customer",
            split_type=split_type,
            start_neighbor=start_neighbor,
            no_return=True,
        )
    )

    """ Filter nodes and experiments into lists """
    edge_index = defaultdict(list)

    for from_type, rel_type, to_type, from_id, to_id in result[0][0]:
        edge_index[
            (
                from_type,
                rel_type.replace("_TRAIN", "").replace("_TEST", "").replace("_VAL", ""),
                to_type,
            )
        ].append((int(from_id), int(to_id)))

    for key, index in edge_index.items():
        if len(index) > 0:
            edge_index[key] = t.tensor(list(map(list, zip(*index))), dtype=t.long)
        else:
            edge_index[key] = t.empty(0, dtype=t.long)

    return edge_index


def get_id_map(db: Database) -> tuple[dict, dict]:
    customers = db.run_match(db.query_all_nodes(node_type="customer"))
    articles = db.run_match(db.query_all_nodes(node_type="article"))

    customer_map = {customer["n"].id: customer["n"]._id for customer in customers}
    article_map = {article["n"].id: article["n"]._id for article in articles}

    return customer_map, article_map
