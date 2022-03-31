from data.types import DataLoaderConfig
from data.data_loader import create_dataloaders
from torch_geometric import seed_everything
import torch
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as PyGData
from model.lightgcn import GNN
import os
import numpy as np
from utils.sample_negative import sample_negative_edges
from typing import Optional
from config import config, Config


def train(
    model: GNN,
    data_mp: PyGData,
    loader: DataLoader,
    opt: Optimizer,
    num_customers: int,
    num_nodes: int,
    device: torch.device,
):
    """
    Main training loop
    args:
       model: the GNN model
       data_mp: message passing edges to use for performing propagation/calculating multi-scale embeddings
       loader: DataLoader that loads in batches of supervision/evaluation edges
       opt: the optimizer
       num_customers: the number of customers in the entire dataset
       num_nodes: the number of nodes (customers + articles) in the entire dataset
       device: whether to run on CPU or GPU
    returns:
       the training loss for this epoch
    """
    total_loss = 0
    total_examples = 0
    model.train()
    i = 0
    for batch in loader:
        print("Iter: ", i)
        i += 1
        del batch.batch
        del batch.ptr  # delete unwanted attributes
        print(batch)

        opt.zero_grad()
        negs = sample_negative_edges(
            batch, num_customers, num_nodes
        )  # sample negative edges
        data_mp, batch, negs = data_mp.to(device), batch.to(device), negs.to(device)
        loss = model.calc_loss(data_mp, batch, negs)
        loss.backward()
        opt.step()

        num_examples = batch.edge_index.shape[1]
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    avg_loss = total_loss / total_examples
    return avg_loss


def test(
    model: GNN,
    data_mp: PyGData,
    loader: DataLoader,
    k: int,
    device: torch.device,
    save_dir: Optional[str],
    epoch: int,
):
    """
    Evaluation loop for validation/testing.
    args:
       model: the GNN model
       data_mp: message passing edges to use for propagation/calculating multi-scale embeddings
       loader: DataLoader that loads in batches of evaluation (i.e., validation or test) edges
       k: value of k to use for recall@k
       device: whether to use CPU or GPU
       save_dir: directory to save multi-scale embeddings for later analysis. If None, doesn't save any embeddings.
       epoch: the number of the current epoch
    returns:
       recall@k for this epoch
    """
    model.eval()
    all_recalls = {}
    with torch.no_grad():
        # Save multi-scale embeddings if save_dir is not None
        data_mp = data_mp.to(device)
        if save_dir is not None:
            embs_to_save = model.gnn_propagation(data_mp.edge_index)
            torch.save(
                embs_to_save, os.path.join(save_dir, f"embeddings_epoch_{epoch}.pt")
            )

        # Run evaluation
        for batch in loader:
            del batch.batch
            del batch.ptr  # delete unwanted attributes

            batch = batch.to(device)
            recalls = model.evaluation(data_mp, batch, k)
            for customer_idx in recalls:
                assert customer_idx not in all_recalls
            all_recalls.update(recalls)
    recall_at_k = np.mean(list(all_recalls.values()))
    return recall_at_k


def run_pipeline(config: Config):
    seed_everything(5)

    (
        train_data,
        val_data,
        test_data,
        customer_id_map,
        article_id_map,
    ) = create_dataloaders(DataLoaderConfig(test_split=0.15, val_split=0.15))
    num_customers = len(customer_id_map)
    num_nodes = len(customer_id_map) + len(article_id_map)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoaders for the supervision/evaluation edges (one each for train/val/test sets)
    train_loader = DataLoader(train_data[0], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data[0], batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_data[0], batch_size=config.batch_size, shuffle=False)

    # Initialize GNN model
    gnn = GNN(
        embedding_dim=config.embedding_dim,
        num_nodes=num_nodes,
        num_customers=num_customers,
        num_layers=config.num_layers,
    ).to(device)

    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)  # using Adam optimizer

    all_train_losses = []  # list of (epoch, training loss)
    all_val_recalls = []  # list of (epoch, validation recall@k)

    # Main training loop
    for epoch in range(config.epochs):
        train_loss = train(
            gnn, train_data[1], train_loader, opt, num_customers, num_nodes, device
        )
        all_train_losses.append((epoch, train_loss))

        if epoch % 5 == 0:
            val_recall = test(
                gnn,
                val_data[1],
                val_loader,
                config.k,
                device,
                config.save_emb_dir,
                epoch,
            )
            all_val_recalls.append((epoch, val_recall))
            print(f"Epoch {epoch}: train loss={train_loss}, val_recall={val_recall}")
        else:
            print(f"Epoch {epoch}: train loss={train_loss}")

    print()

    # Print best validation recall@k value
    best_val_recall = max(all_val_recalls, key=lambda x: x[1])
    print(
        f"Best validation recall@k: {best_val_recall[1]} at epoch {best_val_recall[0]}"
    )

    # Print final recall@k on test set
    test_recall = test(gnn, test_data[1], test_loader, config.k, device, None, 1)
    print(f"Test set recall@k: {test_recall}")


if __name__ == "__main__":
    run_pipeline(config)
