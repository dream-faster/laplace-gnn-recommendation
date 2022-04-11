from data.types import DataLoaderConfig
from data.data_loader_hetero import create_dataloaders
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader
import torch
from config import config, Config

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer

from tqdm import tqdm
import math

from model.encoder_decoder_model import Encoder_Decoder_Model
from utils.loss_functions import weighted_mse_loss


def train(
    dataloader: NeighborLoader, model: Module, optimizer: Optimizer
) -> tuple[float, Module]:
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
    model = Encoder_Decoder_Model(
        hidden_channels=32, metadata=first_batch.metadata()
    ).to(device)

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
