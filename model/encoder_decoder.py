import torch
from data.types import DataLoaderConfig, FeatureInfo
from torch.nn import Linear, Embedding, ModuleList, LayerNorm, BatchNorm2d
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from typing import List, Tuple
from torch import Tensor
from typing import Union, Optional
from torch_geometric.data import Data, HeteroData
from utils.constants import Constants


class GNNEncoder(torch.nn.Module):
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


class EdgeDecoder(torch.nn.Module):
    def __init__(
        self,
        layers: ModuleList,
    ):
        super().__init__()
        self.layers = layers

    def forward(self, z_dict: dict, edge_label_index: dict) -> torch.Tensor:
        customer_index, article_index = edge_label_index
        z = torch.cat(
            [z_dict[Constants.node_user][customer_index], z_dict[Constants.node_item][article_index]],
            dim=-1,
        )
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                z = layer(z)
            else:
                z = layer(z).relu()

        return z.view(-1)


class Encoder_Decoder_Model(torch.nn.Module):
    def __init__(
        self,
        encoder_layers: ModuleList,
        decoder_layers: ModuleList,
        feature_info: FeatureInfo,
        metadata: Tuple[List[str], List[Tuple[str]]],
        embedding: bool,
    ):
        super().__init__()
        self.embedding: bool = embedding
        self.encoder = GNNEncoder(encoder_layers)
        self.encoder = to_hetero(self.encoder, metadata, aggr="sum")
        self.decoder = EdgeDecoder(decoder_layers)
        
        self.encoder_layer_norm_customer = BatchNorm2d(encoder_layers[-1].out_channels)
        self.encoder_layer_norm_article = BatchNorm2d(encoder_layers[-1].out_channels)

        if self.embedding:
            customer_info, article_info = feature_info
            embedding_articles: List[Embedding] = []
            embedding_customers: List[Embedding] = []

            embedding_customers = [
                Embedding(
                    num_embeddings = int(customer_info.num_cat[i] + 1),
                    embedding_dim = int(customer_info.embedding_size[i]),
                    max_norm = 1
                )
                for i in range(customer_info.num_feat)
            ]

            embedding_articles = [
                Embedding(
                    num_embeddings = int(article_info.num_cat[i] + 1),
                    embedding_dim = int(article_info.embedding_size[i]),
                    max_norm = 1
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

        x_dict[Constants.node_user] = torch.cat(embedding_customers, dim=1)
        x_dict[Constants.node_item] = torch.cat(embedding_articles, dim=1)

        return x_dict

    def initialize_encoder_input_size(self, data: HeteroData) -> None:
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        if self.embedding:
            x_dict = self.__embedding(x_dict)

        self.encoder(x_dict, edge_index_dict)

    def forward(
        self, x_dict, edge_index_dict: dict, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        if self.embedding:
            x_dict = self.__embedding(x_dict)
        z_dict = self.encoder(x_dict, edge_index_dict)
        z_dict['customer'] = self.encoder_layer_norm_customer(z_dict['customer'])
        z_dict['article'] = self.encoder_layer_norm_article(z_dict['article'])
        output = self.decoder(z_dict, edge_label_index)
        return output


    def infer(self,  x_dict, edge_index_dict: dict, edge_label_index: torch.Tensor)->Tensor:
        self.eval()
        out = self.forward(x_dict, edge_index_dict, edge_label_index).detach()
        
        # Rebatching by user.
        
        out_per_user = []
        for i, user_index in enumerate(edge_label_index[0]):
            user_id = user_index.item()
            score = out[i].unsqueeze(0)
            if user_id >= len(out_per_user):
                out_per_user.append(score)
            else:
                out_per_user[user_id] = torch.concat([out_per_user[user_id], score])

        max_output_length = max([element.shape[0] for element in out_per_user])
        
        padded_tensors = [F.pad(output, (0,max_output_length-output.shape[0]), "constant", -(1 << 50)) for output in out_per_user]
        
        
        return torch.stack(padded_tensors)
        