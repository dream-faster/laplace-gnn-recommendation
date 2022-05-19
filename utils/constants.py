from typing import Tuple

node_user = "customer"
node_item = "movie"
rel_type = "rated"
rel_rev_type = "rev_rated"


class Constants:
    edge_key = (node_user, rel_type, node_item)
    rev_edge_key = (node_item, rel_rev_type, node_user)
    node_user = node_user
    node_item = node_item
    rel_type = rel_type
    rel_rev_type = rel_rev_type
