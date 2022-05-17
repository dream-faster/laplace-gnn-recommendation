from data.neo4j.neo4j_database import Database
from neo4j.graph import Node, Relationship
import numpy as np


db = Database("bolt://localhost:7687", "neo4j", "password")
node_id = 0

result = db.run_match(
    db.get_n_neighbors(
        node_id=node_id, n_neighbor=2, node_type="Customer", no_return=True
    )
    + db.get_node(node_id=node_id, node_type="Customer", no_return=True)
    + "RETURN n,m,r"
)
nodes = [node for node in result if isinstance(node, Node)]
relationships = [
    relship for list in result for relship in list if isinstance(relship, Relationship)
]


edge_index = [
    (relship.start_node._properties["_id"], relship.end_node._properties["_id"])
    for relship in relationships
]

edge_index = list(set(edge_index))

edge_index_t = list(map(list, zip(*edge_index)))
# node_features= []
print("")
