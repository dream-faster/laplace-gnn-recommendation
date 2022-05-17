import matplotlib.pyplot as plt
import torch as t
from torch import optim
from tqdm import tqdm
from torch_geometric.utils import structured_negative_sampling
from model.lightgcn import LightGCN
from data.lightgcn_loader import create_dataloaders_lightgcn, sample_mini_batch
from utils.metrics_lightgcn import (
    get_metrics_lightgcn,
    bpr_loss,
    create_edges_dict_indexed_by_user,
    make_predictions_for_user,
)

from config import LightGCNConfig, lightgcn_config


# wrapper function to evaluate model
def evaluation(
    model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val
):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (t.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index
    )
    edges = structured_negative_sampling(
        edge_index.to("cpu"),
        num_nodes=t.max(edge_index[1]).to("cpu"),
        contains_neg_self_loops=False,
    )
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = (
        users_emb_final[user_indices],
        users_emb_0[user_indices],
    )
    pos_items_emb_final, pos_items_emb_0 = (
        items_emb_final[pos_item_indices],
        items_emb_0[pos_item_indices],
    )
    neg_items_emb_final, neg_items_emb_0 = (
        items_emb_final[neg_item_indices],
        items_emb_0[neg_item_indices],
    )

    loss = bpr_loss(
        users_emb_final,
        users_emb_0,
        pos_items_emb_final,
        pos_items_emb_0,
        neg_items_emb_final,
        neg_items_emb_0,
        lambda_val,
    ).item()

    recall, precision, ndcg = get_metrics_lightgcn(
        model, edge_index, exclude_edge_indices, k
    )

    return loss, recall, precision, ndcg


def train(config: LightGCNConfig):
    config.print()
    (
        train_sparse_edge_index,
        val_sparse_edge_index,
        test_sparse_edge_index,
        train_edge_index,
        val_edge_index,
        test_edge_index,
        edge_index,
        num_users,
        num_articles,
    ) = create_dataloaders_lightgcn()

    # setup
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Using device {device}.")

    model = LightGCN(
        num_users,
        num_articles,
        embedding_dim=config.hidden_layer_size,
        num_iterations=config.num_iterations,
    )
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    edge_index = edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    train_sparse_edge_index = train_sparse_edge_index.to(device)

    val_edge_index = val_edge_index.to(device)
    val_sparse_edge_index = val_sparse_edge_index.to(device)

    # training loop
    train_losses = []
    val_losses = []

    loop_obj = tqdm(range(0, config.epochs))
    for iter in loop_obj:
        # forward propagation
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
            train_sparse_edge_index
        )

        # mini batching
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
            config.batch_size, train_edge_index
        )
        user_indices, pos_item_indices, neg_item_indices = (
            user_indices.to(device),
            pos_item_indices.to(device),
            neg_item_indices.to(device),
        )
        users_emb_final, users_emb_0 = (
            users_emb_final[user_indices],
            users_emb_0[user_indices],
        )
        pos_items_emb_final, pos_items_emb_0 = (
            items_emb_final[pos_item_indices],
            items_emb_0[pos_item_indices],
        )
        neg_items_emb_final, neg_items_emb_0 = (
            items_emb_final[neg_item_indices],
            items_emb_0[neg_item_indices],
        )

        # loss computation
        train_loss = bpr_loss(
            users_emb_final,
            users_emb_0,
            pos_items_emb_final,
            pos_items_emb_0,
            neg_items_emb_final,
            neg_items_emb_0,
            config.Lambda,
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if iter % config.eval_every == 0:
            model.eval()
            val_loss, recall, precision, ndcg = evaluation(
                model,
                val_edge_index,
                val_sparse_edge_index,
                [train_edge_index],
                config.k,
                config.Lambda,
            )
            print(
                f"[Iter {iter}/{config.epochs}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{config.k}: {round(recall, 5)}, val_precision@{config.k}: {round(precision, 5)}, val_ndcg@{config.k}: {round(ndcg, 5)}"
            )
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)
            model.train()

        if iter % config.lr_decay_every == 0 and iter != 0:
            scheduler.step()

    if config.show_graph:
        iters = [iter * config.eval_every for iter in range(len(train_losses))]
        plt.plot(iters, train_losses, label="train")
        plt.plot(iters, val_losses, label="validation")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("training and validation loss curves")
        plt.legend()
        plt.show()

    # evaluate on test set
    model.eval()
    test_edge_index = test_edge_index.to(device)
    test_sparse_edge_index = test_sparse_edge_index.to(device)

    test_loss, test_recall, test_precision, test_ndcg = evaluation(
        model,
        test_edge_index,
        test_sparse_edge_index,
        [train_edge_index, val_edge_index],
        config.k,
        config.Lambda,
    )

    print(
        f"[test_loss: {round(test_loss, 5)}, test_recall@{config.k}: {round(test_recall, 5)}, test_precision@{config.k}: {round(test_precision, 5)}, test_ndcg@{config.k}: {round(test_ndcg, 5)}"
    )

    # Save predictions for the matcher
    model.eval()
    pos_items_per_user = create_edges_dict_indexed_by_user(edge_index)
    top_items_per_user = {}
    for user in tqdm(range(0, num_users)):
        top_items_per_user[user] = make_predictions_for_user(
            model.users_emb.weight,
            model.items_emb.weight,
            user,
            pos_items_per_user,
            config.num_recommendations,
        )
    t.save(top_items_per_user, "data/derived/lightgcn_output.pt")

    save_scores(model)


def save_scores(model: LightGCN):
    print("| Saving the user and article final embeddings...")
    t.save(model.users_emb.weight, "data/derived/users_emb_final_lightgcn.pt")
    t.save(model.items_emb.weight, "data/derived/items_emb_final_lightgcn.pt")


if __name__ == "__main__":
    train(lightgcn_config)
