from neo4j.graph import Node, Relationship
from data.neo4j.neo4j_database import Database


def get_neighborhood(
    db: Database, node_id: int, n_neighbor: int, split_type: str
) -> list:
    node_id = 0

    result = db.run_match(
        db.get_n_neighbors(
            node_id=node_id, n_neighbor=n_neighbor, node_type="Customer", no_return=True
        )
        + db.get_node(node_id=node_id, node_type="Customer", no_return=True)
        + "RETURN collect(n), collect(m), collect(r)"
    )

    """ Collect nodes and experiments into lists """
    user_node = result[0][0]
    neighbor_nodes = result[1]
    relationships = [relship for list in result[2] for relship in list]

    """ Filter nodes and experiments into lists """
    edge_index = [
        (
            int(relship.start_node._properties["_id"]),
            int(relship.end_node._properties["_id"]),
        )
        for relship in relationships
        if int(relship._properties[split_type + "_mask"]) == 1
    ]

    edge_index = list(set(edge_index))
    edge_index_t = list(map(list, zip(*edge_index)))

    return edge_index_t
