from typing import Tuple
from data.types import ArticleColumn

node_user = "customer"
node_item = "article"
rel_type = "buys"
rel_rev_type = "rev_buys"
node_extra = ArticleColumn.ColourGroupCode.value
rel_type_extra = "has_color"


class Constants:
    edge_key = (node_user, rel_type, node_item)
    rev_edge_key = (node_item, rel_rev_type, node_user)
    edge_key_extra = (node_item, rel_type_extra, node_extra)
    node_user = node_user
    node_item = node_item
    rel_type = rel_type
    rel_rev_type = rel_rev_type
    rel_type_extra = rel_type_extra
    node_extra = node_extra
