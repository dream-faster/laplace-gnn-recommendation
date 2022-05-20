from neo4j.graph import Node, Relationship
from data.neo4j.neo4j_database import Database
from utils.constants import Constants


def get_neighborhood(
    db: Database, node_id: int, n_neighbor: int, split_type: str
) -> list:
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
    edge_index = [
        (int(res[1]), int(res[2]))
        for res in result[0][0]
        if res[0] == Constants.node_user
    ]
    edge_index.extend(
        [
            (int(res[2]), int(res[1]))
            for res in result[0][0]
            if res[0] == Constants.node_item
        ]
    )
    edge_index.extend(
        [
            (int(res[2]), int(res[1]))
            for res in result[0][0]
            if res[0] == Constants.node_extra
        ]
    )

    edge_index = list(set(edge_index))
    edge_index_t = list(map(list, zip(*edge_index)))

    return edge_index_t


def get_id_map(db: Database) -> tuple[dict, dict]:
    customers = db.run_match(db.query_all_nodes(node_type="Customer"))
    articles = db.run_match(db.query_all_nodes(node_type="Article"))

    customer_map = {customer["n"].id: customer["n"]._id for customer in customers}
    article_map = {article["n"].id: article["n"]._id for article in articles}

    return customer_map, article_map
