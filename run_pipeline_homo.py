from data.types import DataLoaderConfig, FeatureInfo, PipelineConst
from data.data_loader_homo import create_dataloaders, create_datasets
from torch_geometric import seed_everything
import torch
from config import config, Config


import torch
import torch.nn.functional as F
from torch.nn import Module

from tqdm import tqdm
import math
from torch.optim import Optimizer

from model.encoder_decoder_hetero import Encoder_Decoder_Model
from utils.loss_functions import weighted_mse_loss
from utils.get_info import get_feature_info


def train(data, model: Module, optimizer: Optimizer) -> tuple[float, Module]:
    model.train()
    optimizer.zero_grad()
    pred: torch.Tensor = model(
        data.x_dict,
        data.edge_index_dict,
        data.edge_label_index,
    )
    target = data.edge_label
    loss: torch.Tensor = weighted_mse_loss(pred, target, None)
    loss.backward()
    optimizer.step()
    return float(loss), model


@torch.no_grad()
def test(data, model):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, data.edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data.edge_label.float()
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
        full_data,
    ) = create_datasets(
        DataLoaderConfig(test_split=0.15, val_split=0.15, batch_size=32)
    )
    assert torch.max(train_loader.edge_stores[0].edge_index) <= train_loader.num_nodes

    print("| Creating Model...")
    feature_info = get_feature_info(full_data, config.type)
    model = Encoder_Decoder_Model(
        hidden_channels=32, feature_info=feature_info, metadata=train_loader.metadata()
    ).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    print("| Lazy Initialization of Model...")
    with torch.no_grad():
        model.initialize_encoder_input_size(
            train_loader.x_dict, train_loader.edge_index_dict
        )

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
