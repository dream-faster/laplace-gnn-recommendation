from .type import Matcher
import torch as t
from torch import Tensor


class LightGCNMatcher(Matcher):
    def __init__(self, k: int):
        self.scores = get_scores()
        self.k = k

    def get_matches(self, user_id: int) -> t.Tensor:
        candidate_scores, candidate_ids = t.topk(self.scores[user_id], k=self.k)
        return candidate_ids


def get_scores() -> Tensor:
    item_embeddings: Tensor = t.load("data/derived/items_emb_final_lightgcn.pt")
    user_embeddings: Tensor = t.load("data/derived/users_emb_final_lightgcn.pt")
    return item_embeddings @ user_embeddings.T
