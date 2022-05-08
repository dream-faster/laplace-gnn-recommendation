from tqdm import tqdm
import torch
import numpy as np

from torch_geometric import seed_everything

from config import link_pred_config, Config
from model.encoder_decoder import Encoder_Decoder_Model

from utils.get_info import get_feature_info
from data.data_loader import create_dataloaders
from single_epoch import epoch_with_dataloader
from model.layers import get_linear_layers, get_SAGEConv_layers
from single_epoch import test

from reporting.wandb import setup_config


def run_pipeline(config: Config, with_wandb: bool = False):
    config.print()
    wandb, config = setup_config("Fashion-Recomm-GNN", with_wandb, config)

    print("| Seeding everything...")
    seed_everything(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert (
        config.k <= config.dataloader_config.candidate_pool_size
    ), "k must be smaller than candidate_pool_size"

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
        encoder_layers=get_SAGEConv_layers(
            num_layers=config.num_gnn_layers,
            hidden_channels=config.hidden_layer_size,
            out_channels=config.encoder_layer_output_size,
            agg_type=config.conv_agg_type,
        ),
        decoder_layers=get_linear_layers(
            num_layers=config.num_linear_layers,
            in_channels=config.encoder_layer_output_size * 2,
            hidden_channels=config.hidden_layer_size,
            out_channels=1,
        ),
        feature_info=get_feature_info(full_data),
        metadata=next(iter(train_loader)).metadata(),
        embedding=True,
        heterogeneous_prop_agg_type=config.heterogeneous_prop_agg_type,
    ).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    print("| Lazy Initialization of Model...")
    with torch.no_grad():
        model.initialize_encoder_input_size(next(iter(train_loader)).to(device))

    print("| Defining Optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("| Training Model...")
    old_val_precision = -1
    loop_obj = tqdm(range(0, config.epochs))
    for epoch in loop_obj:
        val_precision = epoch_with_dataloader(
            model,
            optimizer,
            train_loader,
            val_loader,
            test_loader,
            epoch_id=epoch,
            config=config,
        )

        # We should save our model if validation precision starts decreasing (it starts to overfit)
        if val_precision >= old_val_precision:
            old_val_precision = val_precision.copy()
        else:
            print("| Saving Best Generalized Model...")
            torch.save(model.state_dict(), f"model/saved/model_final.pt")
            old_val_precision = (
                -1
            )  # We should only save it at the inflection point from decreasing one step

        if epoch % max(1, int(config.epochs * config.save_every)) == 0:
            print("| Saving Model at a regular interval...")
            torch.save(model.state_dict(), f"model/saved/model_{epoch:03d}.pt")

    # Testing loop
    test_recalls, test_precisions = [], []
    test_loop = tqdm(iter(test_loader), colour="blue")
    for i, data in enumerate(test_loop):
        if config.evaluate_break_at and i == config.evaluate_break_at:
            break
        test_loop.set_description("TEST")
        test_recall, test_precision = test(data.to(device), model, [], k=config.k)
        test_recalls.append(test_recall)
        test_precisions.append(test_precision)
        test_loop.set_postfix_str(
            f"Recall: {np.mean(test_recalls):.4f} | Precision: {np.mean(test_precisions):.4f}"
        )


if __name__ == "__main__":
    run_pipeline(link_pred_config)
