from data.types import DataLoaderConfig
from data.data_loader import create_dataloaders
from torch_geometric import seed_everything
import torch
from typing import Optional
from config import config, Config

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import SAGEConv, to_hetero


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict["user"][row], z_dict["movie"][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data["user", "movie"].edge_label_index,
    )
    target = train_data["user", "movie"].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(
        data.x_dict, data.edge_index_dict, data["user", "movie"].edge_label_index
    )
    pred = pred.clamp(min=0, max=5)
    target = data["user", "movie"].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


def run_pipeline(config: Config):
    seed_everything(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_data,
        val_data,
        test_data,
        customer_id_map,
        article_id_map,
    ) = create_dataloaders(DataLoaderConfig(test_split=0.15, val_split=0.15))

    model = Model(hidden_channels=32).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 301):
        loss = train()
        train_rmse = test(train_data)
        val_rmse = test(val_data)
        test_rmse = test(test_data)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
            f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
        )


if __name__ == "__main__":
    run_pipeline(config)
