import torch as t
from data.types import FeatureInfo
from torch.nn import Linear, Embedding, ModuleList, LayerNorm, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from typing import List, Tuple
from torch import Tensor
from typing import Union, Optional
from torch_geometric.data import Data, HeteroData
from utils.constants import Constants
from utils.tensor import padded_stack


class GNNEncoder(t.nn.Module):
    def __init__(
        self,
        layers: ModuleList,
    ):
        super().__init__()
        self.layers = layers

    def forward(self, x, edge_index):
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index).relu()

        return x


class EdgeDecoder(t.nn.Module):
    def __init__(
        self,
        layers: ModuleList,
    ):
        super().__init__()
        self.layers = layers

    def forward(self, z_dict: dict, edge_label_index: dict) -> t.Tensor:
        customer_index, article_index = edge_label_index
        z = t.cat(
            [
                z_dict[Constants.node_user][customer_index],
                z_dict[Constants.node_item][article_index],
            ],
            dim=-1,
        )
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                z = layer(z)
            else:
                z = layer(z).relu()

        return z.view(-1)


class Encoder_Decoder_Model(t.nn.Module):
    def __init__(
        self,
        encoder_layers: ModuleList,
        decoder_layers: ModuleList,
        feature_info: FeatureInfo,
        metadata: Tuple[List[str], List[Tuple[str]]],
        embedding: bool,
        heterogeneous_prop_agg_type: str,  # "sum", "mean", "min", "max", "mul"
    ):
        super().__init__()
        self.embedding: bool = embedding
        self.encoder = GNNEncoder(encoder_layers)
        self.encoder = to_hetero(
            self.encoder, metadata, aggr=heterogeneous_prop_agg_type
        )
        self.decoder = EdgeDecoder(decoder_layers)

        self.encoder_layer_norm_customer = BatchNorm1d(encoder_layers[-1].out_channels)
        self.encoder_layer_norm_article = BatchNorm1d(encoder_layers[-1].out_channels)

        if self.embedding:
            customer_info, article_info = feature_info
            embedding_articles: List[Embedding] = []
            embedding_customers: List[Embedding] = []

            embedding_customers = [
                Embedding(
                    num_embeddings=int(customer_info.num_cat[i] + 1),
                    embedding_dim=int(customer_info.embedding_size[i]),
                    max_norm=1,
                )
                for i in range(customer_info.num_feat)
            ]

            embedding_articles = [
                Embedding(
                    num_embeddings=int(article_info.num_cat[i] + 1),
                    embedding_dim=int(article_info.embedding_size[i]),
                    max_norm=1,
                )
                for i in range(article_info.num_feat)
            ]

            self.embedding_customers = ModuleList(embedding_customers)
            self.embedding_articles = ModuleList(embedding_articles)

    def __embedding(self, x_dict: dict) -> dict:
        customer_features, article_features = (
            x_dict[Constants.node_user].long(),
            x_dict[Constants.node_item].long(),
        )
        embedding_customers, embedding_articles = [], []
        for i, embedding_layer in enumerate(self.embedding_customers):
            embedding_customers.append(embedding_layer(customer_features[:, i]))

        for i, embedding_layer in enumerate(self.embedding_articles):
            embedding_articles.append(embedding_layer(article_features[:, i]))

        x_dict[Constants.node_user] = t.cat(embedding_customers, dim=1)
        x_dict[Constants.node_item] = t.cat(embedding_articles, dim=1)

        return x_dict

    def initialize_encoder_input_size(self, data: HeteroData) -> None:
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        if self.embedding:
            x_dict = self.__embedding(x_dict)

        self.encoder(x_dict, edge_index_dict)

    def forward(
        self, x_dict, edge_index_dict: dict, edge_label_index: t.Tensor
    ) -> t.Tensor:
        if self.embedding:
            x_dict = self.__embedding(x_dict)
        z_dict = self.encoder(x_dict, edge_index_dict)
        z_dict[Constants.node_user] = self.encoder_layer_norm_customer(
            z_dict[Constants.node_user]
        )
        z_dict[Constants.node_item] = self.encoder_layer_norm_article(
            z_dict[Constants.node_item]
        )
        output = self.decoder(z_dict, edge_label_index)
        return output

    def infer(
        self, x_dict, edge_index_dict: dict, edge_label_index: t.Tensor
    ) -> Tensor:
        self.eval()
        out = self.forward(x_dict, edge_index_dict, edge_label_index).detach()

        # Rebatching by user.
        users = edge_label_index[0].unique(sorted=True)
        users = t.bucketize(users, users)
        out_per_user = [out[edge_label_index[0] == user] for user in users]
        return padded_stack(out_per_user, value=-(1 << 50))
