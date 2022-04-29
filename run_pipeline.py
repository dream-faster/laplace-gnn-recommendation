from tqdm import tqdm
import torch

from torch_geometric import seed_everything

from config import link_pred_config, Config
from model.encoder_decoder import Encoder_Decoder_Model

from utils.get_info import get_feature_info
from data.data_loader import create_dataloaders
from single_epoch import epoch_with_dataloader


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
    ) = create_dataloaders(config.dataloader_config)

    print("| Creating Model...")
    model = Encoder_Decoder_Model(
        hidden_channels=config.hidden_layer_size,
        feature_info=get_feature_info(full_data),
        metadata=full_data.metadata(),
        embedding=True,
    ).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    print("| Lazy Initialization of Model...")
    with torch.no_grad():
        model.initialize_encoder_input_size(next(iter(train_loader)))

    print("| Defining Optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("| Training Model...")
    loop_obj = tqdm(range(0, config.epochs))
    for epoch in loop_obj:
        loss, val_rmse, test_rmse = epoch_with_dataloader(
            model, optimizer, train_loader, val_loader, test_loader
        )
        torch.save(model.state_dict(), f"model/saved/model_{epoch:03d}.pt")


if __name__ == "__main__":
    run_pipeline(link_pred_config)
