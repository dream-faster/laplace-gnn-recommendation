import torch
from torch import Tensor
from data.types import DataLoaderConfig, FeatureInfo
from torch.nn import Linear, Embedding, ModuleList
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z: Tensor, edge_label_index: Tensor) -> torch.Tensor:
        z = z[torch.flatten(edge_label_index)]

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Encoder_Decoder_Model_Homo(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        feature_info: FeatureInfo,
        embedding: bool = False,
    ):
        super().__init__()
        self.embedding: bool = embedding
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

        if self.embedding:
            customer_info, article_info = feature_info
            embedding_articles: list[Embedding] = []
            embedding_customers: list[Embedding] = []

            for i in range(article_info.num_feat):
                embedding_customers.append(
                    Embedding(
                        int(customer_info.num_cat[i] + 1),
                        int(customer_info.embedding_size[i]),
                    )
                )
            for i in range(article_info.num_feat):
                embedding_articles.append(
                    Embedding(
                        int(article_info.num_cat[i] + 1),
                        int(article_info.embedding_size[i]),
                    )
                )

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

        x_dict["customer"] = torch.cat(embedding_customers, dim=0)
        x_dict["article"] = torch.cat(embedding_articles, dim=0)

        return x_dict

    def initialize_encoder_input_size(self, data: Data) -> None:
        x, edge_index = data.x, data.edge_index
        if self.embedding:
            x = self.__embedding(x)

        self.encoder(x, edge_index)

    def forward(
        self, x_dict, edge_index_dict: dict, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        if self.embedding:
            x_dict = self.__embedding(x_dict)
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
