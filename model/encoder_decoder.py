import torch
from data.types import DataLoaderConfig, FeatureInfo
from torch.nn import Linear, Embedding, ModuleList
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from typing import List, Tuple


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), int(hidden_channels * 0.6))
        self.conv3 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, int(hidden_channels * 0.5))
        self.lin3 = Linear(int(hidden_channels * 0.5), 1)

    def forward(self, z_dict: dict, edge_label_index: dict) -> torch.Tensor:
        customer_index, article_index = edge_label_index
        z = torch.cat(
            [z_dict["customer"][customer_index], z_dict["article"][article_index]],
            dim=-1,
        )

        z = self.lin1(z).relu()
        z = self.lin2(z).relu()
        z = self.lin3(z)
        return z.view(-1)


class Encoder_Decoder_Model(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        feature_info: FeatureInfo,
        metadata: Tuple[List[str], List[Tuple[str]]],
        embedding: bool,
    ):
        super().__init__()
        self.embedding: bool = embedding
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels)

        if self.embedding:
            customer_info, article_info = feature_info
            embedding_articles: List[Embedding] = []
            embedding_customers: List[Embedding] = []

            embedding_customers = [
                Embedding(
                    int(customer_info.num_cat[i] + 1),
                    int(customer_info.embedding_size[i]),
                )
                for i in range(customer_info.num_feat)
            ]

            embedding_articles = [
                Embedding(
                    int(article_info.num_cat[i] + 1),
                    int(article_info.embedding_size[i]),
                )
                for i in range(article_info.num_feat)
            ]

            self.embedding_customers = ModuleList(embedding_customers)
            self.embedding_articles = ModuleList(embedding_articles)

    def __embedding(self, x_dict: dict) -> dict:
        customer_features, article_features = (
            x_dict["customer"].long(),
            x_dict["article"].long(),
        )
        embedding_customers, embedding_articles = [], []
        for i, embedding_layer in enumerate(self.embedding_customers):
            embedding_customers.append(embedding_layer(customer_features[:, i]))

        for i, embedding_layer in enumerate(self.embedding_articles):
            embedding_articles.append(embedding_layer(article_features[:, i]))

        x_dict["customer"] = torch.cat(embedding_customers, dim=1)
        x_dict["article"] = torch.cat(embedding_articles, dim=1)

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
        return self.decoder(z_dict, edge_label_index)
