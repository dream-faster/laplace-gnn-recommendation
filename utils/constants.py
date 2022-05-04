from typing import Tuple


class Constants:
    edge_key: Tuple[str, str, str] = ("customer", "buys", "article")
    rev_edge_key: Tuple[str, str, str] = ("article", "rev_buys", "customer")
    node_user: str = "customer"
    node_item: str = "article"
