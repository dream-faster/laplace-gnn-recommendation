from neo4j.graph import Node, Relationship
from data.neo4j.neo4j_database import Database


def get_neighborhood(
    db: Database, node_id: int, n_neighbor: int, split_type: str
) -> list:
    result = db.run_match(
        db.query_n_neighbors(
            node_id=node_id,
            n_neighbor=n_neighbor,
            node_type="Customer",
            split_type=split_type,
            no_return=False,
        )
    )

    """ Filter nodes and experiments into lists """
    edge_index = [
        (
            int(relship["startnode(rel)._id"]),
            int(relship["endnode(rel)._id"]),
        )
        for relship in result
    ]

    edge_index = list(set(edge_index))
    edge_index_t = list(map(list, zip(*edge_index)))

    return edge_index_t
