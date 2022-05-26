import torch as t
import torch_geometric
from data.types import FeatureInfo
from torch.nn import Linear, Embedding, ModuleList, LayerNorm, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_adj
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
        p_dropout_edges: Optional[float],
        p_dropout_features: Optional[float],
    ):
        super().__init__()
        self.layers = layers
        self.p_dropout_edges = p_dropout_edges
        self.p_dropout_features = p_dropout_features

    def forward(self, x, edge_index):
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                x = layer(x, edge_index)
            else:
                # if self.p_dropout_edges is not None:
                #     edge_index, _ = dropout_adj(
                #         edge_index,
                #         p=self.p_dropout_edges,
                #         force_undirected=True,
                #         training=self.training,
                #     )
                if self.p_dropout_features is not None:
                    x = F.dropout(x, p=self.p_dropout_features, training=self.training)

                x = layer(x, edge_index).relu()

        return x


class EdgeDecoder(t.nn.Module):
    def __init__(self, layers: ModuleList, p_dropout_features: Optional[float]):
        super().__init__()
        self.layers = layers
        self.p_dropout_features = p_dropout_features

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
                if self.p_dropout_features is not None:
                    z = F.dropout(z, p=self.p_dropout_features, training=self.training)
                z = layer(z).relu()

        return z.view(-1)


class Encoder_Decoder_Model(t.nn.Module):
    def __init__(
        self,
        encoder_layers: ModuleList,
        decoder_layers: ModuleList,
        feature_info: dict[FeatureInfo],
        metadata: Tuple[List[str], List[Tuple[str]]],
        embedding: bool,
        heterogeneous_prop_agg_type: str,  # "sum", "mean", "min", "max", "mul",
        batch_normalize: bool,
        p_dropout_edges: Optional[float],
        p_dropout_features: Optional[float],
    ):
        super().__init__()
        self.embedding = embedding
        self.batch_normalize = batch_normalize

        self.encoder = GNNEncoder(encoder_layers, p_dropout_edges, p_dropout_features)
        self.encoder = to_hetero(
            self.encoder, metadata, aggr=heterogeneous_prop_agg_type
        )
        self.decoder = EdgeDecoder(decoder_layers, p_dropout_features)

        self.encoder_layer_norm_customer = BatchNorm1d(encoder_layers[-1].out_channels)
        self.encoder_layer_norm_article = BatchNorm1d(encoder_layers[-1].out_channels)

        self.embedding_layers = dict()

        if self.embedding:
            for key, item in feature_info.items():
                self.embedding_layers[key] = ModuleList(
                    [
                        Embedding(
                            num_embeddings=int(item.num_cat[i] + 1),
                            embedding_dim=int(item.embedding_size[i]),
                            max_norm=1,
                        )
                        for i in range(item.num_feat)
                    ]
                )

    def __embedding(self, x_dict: dict) -> dict:

        for key, item in self.embedding_layers.items():
            features = x_dict[key]
            embedding = [
                embedding_layer(features[:, i])
                for i, embedding_layer in enumerate(item)
            ]
            x_dict[key] = t.cat(embedding, dim=1)

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

        if self.batch_normalize:
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
        out_per_user = [out[edge_label_index[0] == user] for user in users]
        return padded_stack(out_per_user, value=-(1 << 50))
