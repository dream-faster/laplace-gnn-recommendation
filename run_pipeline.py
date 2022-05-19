from tqdm import tqdm
import torch as t
import numpy as np

from torch_geometric import seed_everything

from config import link_pred_config, Config
from model.encoder_decoder import Encoder_Decoder_Model

from utils.get_info import get_feature_info
from data.data_loader import create_dataloaders
from training import test_with_dataloader, train_with_dataloader
from model.layers import get_linear_layers, get_SAGEConv_layers

from reporting.wandb import setup_config, report_results
from reporting.types import (
    Stats,
    ContinousStatsVal,
    ContinousStatsTest,
    ContinousStatsTrain,
)


def run_pipeline(config: Config) -> Stats:
    config.print()
    config.check_validity()
    wandb, config = setup_config("Fashion-Recomm-GNN", config.wandb_enabled, config)

    print("| Seeding everything...")
    seed_everything(5)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    assert (
        config.k * 2 <= config.candidate_pool_size
    ), "k must be smaller than candidate_pool_size"  # we always have more than one matcher

    print("| Creating Datasets...")
    (
        train_loader,
        val_loader,
        test_loader,
        customer_id_map,
        article_id_map,
        full_data,
    ) = create_dataloaders(config)

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
        batch_normalize=config.batch_norm,
        p_dropout_edges=config.p_dropout_edges,
        p_dropout_features=config.p_dropout_features,
    ).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    print("| Lazy Initialization of Model...")
    with t.no_grad():
        model.initialize_encoder_input_size(next(iter(train_loader)).to(device))

    print("| Defining Optimizer...")
    optimizer = t.optim.Adam(model.parameters(), lr=config.learning_rate)

    print("| Training Model...")
    old_val_precision = -1
    for epoch in range(0, config.epochs):
        losses = train_with_dataloader(model, optimizer, train_loader, epoch, device)

        report_results(
            output_stats=ContinousStatsTrain(
                type="train", loss=np.mean(losses), epoch=epoch
            ),
            wandb=wandb,
            final=False,
        )

        if epoch % config.eval_every == 0 and epoch != 0:
            val_precision, val_recall = test_with_dataloader(
                "VAL",
                model,
                val_loader,
                device,
                k=config.k,
                break_at=config.evaluate_break_at,
            )

            # We should save our model if validation precision starts decreasing (it starts to overfit)
            if val_precision >= old_val_precision:
                old_val_precision = val_precision
            else:
                print("| Saving Best Generalized Model...")
                t.save(model.state_dict(), f"model/saved/model_final.pt")
                old_val_precision = (
                    -1
                )  # We should only save it at the inflection point from decreasing one step

            report_results(
                output_stats=ContinousStatsVal(
                    type="val",
                    recall_val=val_recall,
                    precision_val=val_precision,
                    epoch=epoch,
                ),
                wandb=wandb,
                final=False,
            )

        if epoch % max(1, int(config.epochs * config.save_every)) == 0:
            print("| Saving Model at a regular interval...")
            t.save(model.state_dict(), f"model/saved/model_{epoch:03d}.pt")

    test_recall, test_precision = test_with_dataloader(
        "TEST",
        model,
        test_loader,
        device,
        k=config.k,
        break_at=config.evaluate_break_at,
    )

    report_results(
        output_stats=ContinousStatsTest(
            type="test",
            recall_test=test_recall,
            precision_test=test_precision,
        ),
        wandb=wandb,
        final=True,
    )
    return Stats(
        loss=np.mean(losses),
        recall_val=val_recall,
        recall_test=test_recall,
        precision_val=val_precision,
        precision_test=test_precision,
    )


if __name__ == "__main__":
    run_pipeline(link_pred_config)
