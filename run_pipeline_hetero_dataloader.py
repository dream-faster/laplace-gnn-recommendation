from data.types import DataLoaderConfig
from data.data_loader_hetero import create_dataloaders, create_datasets
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader
import torch
from typing import Optional
from config import config, Config

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import SAGEConv, to_hetero
from tqdm import tqdm
import math


def weighted_mse_loss(pred, target, weight=None):
    weight = 1.0 if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


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
        z = torch.cat([z_dict["customer"][row], z_dict["article"][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


def train(
    dataloader: NeighborLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[float, torch.nn.Module]:
    model.train()
    loss = torch.zeros(0)
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        pred = model(
            data.x_dict,
            data.edge_index_dict,
            data["customer", "article"].edge_label_index,
        )
        target = data["customer", "article"].edge_label
        loss = weighted_mse_loss(pred, target, None)
        loss.backward()
        optimizer.step()
    return float(loss), model


@torch.no_grad()
def test(data, model):
    model.eval()
    pred = model(
        data.x_dict, data.edge_index_dict, data["customer", "article"].edge_label_index
    )
    pred = pred.clamp(min=0, max=5)
    target = data["customer", "article"].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse), model


def run_pipeline(config: Config):
    print("| Seeding everything...")
    seed_everything(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("| Creating Datasets...")
    (
        train_loader,
        val_loader,
        test_loader,
        customer_id_map,
        article_id_map,
    ) = create_dataloaders(
        DataLoaderConfig(test_split=0.01, val_split=0.01, batch_size=128)
    )

    print("| Creating Model...")
    first_batch = next(iter(train_loader))
    model = Model(hidden_channels=32, metadata=first_batch.metadata()).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    print("| Lazy Initialization of Model...")
    with torch.no_grad():
        model.encoder(first_batch.x_dict, first_batch.edge_index_dict)

    print("| Defining Optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("| Training Model...")
    num_epochs = config.epochs
    loop_obj = tqdm(range(0, num_epochs))
    for epoch in loop_obj:
        loss, model = train(train_loader, model, optimizer)
        train_rmse, model = test(train_loader, model)
        val_rmse, model = test(val_loader, model)
        test_rmse, model = test(test_loader, model)

        loop_obj.set_postfix_str(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
            f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
        )
        if epoch % math.floor(num_epochs / 3) == 0:
            torch.save(model.state_dict(), f"model/saved/model_{epoch:03d}.pt")
            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
                f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
            )


if __name__ == "__main__":
    run_pipeline(config)
