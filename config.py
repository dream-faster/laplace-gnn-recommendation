from dataclasses import dataclass
from typing import Optional, Union
from data.types import PipelineConst


@dataclass
class Config:
    epochs: int  # number of training epochs
    k: int  # value of k for recall@k. It is important to set this to a reasonable value!
    num_layers: int  # number of LightGCN layers (i.e., number of hops to consider during propagation)
    batch_size: int  # batch size. refers to the # of customers in the batch (each will come with all of its edges)
    embedding_dim: int  # dimension to use for the customer/article embeddings
    save_emb_dir: Optional[
        str
    ]  # path to save multi-scale embeddings during test(). If None, will not save any embeddings
    type: Union[PipelineConst.heterogenous, PipelineConst.homogenous]  # type of graph we use

config = Config(
    epochs=1500, k=12, num_layers=3, batch_size=1, embedding_dim=64, save_emb_dir=None, type=PipelineConst.homogenous
)
