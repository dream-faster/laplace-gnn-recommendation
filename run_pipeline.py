from data.types import DataLoaderConfig
from data.data_loader import run_dataloader
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


def train(
    model: GNN,
    data_mp: PyGData,
    loader: DataLoader,
    opt: Optimizer,
    num_customers: int,
    num_nodes: int,
    device: str,
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
    for batch in loader:
        del batch.batch
        del batch.ptr  # delete unwanted attributes

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
    device: str,
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


def run_pipeline():
    seed_everything(5)

    config = DataLoaderConfig(test_split=0.15, val_split=0.15)
    train_data, val_data, test_data = run_dataloader(config)

    # Training hyperparameters
    epochs = 300  # number of training epochs
    k = 250  # value of k for recall@k. It is important to set this to a reasonable value!
    num_layers = 3  # number of LightGCN layers (i.e., number of hops to consider during propagation)
    batch_size = 2048  # batch size. refers to the # of customers in the batch (each will come with all of its edges)
    embedding_dim = 64  # dimension to use for the playlist/song embeddings
    save_emb_dir = None  # path to save multi-scale embeddings during test(). If None, will not save any embeddings

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoaders for the supervision/evaluation edges (one each for train/val/test sets)
    train_loader = DataLoader(train_data[0], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data[0], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data[0], batch_size=batch_size, shuffle=False)

    # Initialize GNN model
    gnn = GNN(
        embedding_dim=embedding_dim,
        num_nodes=data.num_nodes,
        num_customers=num_playlists,
        num_layers=num_layers,
    ).to(device)

    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)  # using Adam optimizer

    all_train_losses = []  # list of (epoch, training loss)
    all_val_recalls = []  # list of (epoch, validation recall@k)

    # Main training loop
    for epoch in range(epochs):
        train_loss = train(
            gnn, train_mp, train_loader, opt, num_customers, num_nodes, device
        )
        all_train_losses.append((epoch, train_loss))

        if epoch % 5 == 0:
            val_recall = test(gnn, val_mp, val_loader, k, device, save_emb_dir, epoch)
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
    test_recall = test(gnn, test_mp, test_loader, k, device, None, None)
    print(f"Test set recall@k: {test_recall}")


if __name__ == "__main__":
    run_pipeline()
