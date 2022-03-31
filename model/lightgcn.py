import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from utils.metrics import recall_at_k
from torch_geometric.data import Data as PyGData


class GNN(torch.nn.Module):
    """
    Overall graph neural network. Consists of learnable user/item embeddings
    and LightGCN layers.
    """

    def __init__(
        self, embedding_dim: int, num_nodes: int, num_customers: int, num_layers: int
    ):
        super(GNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.num_customers = num_customers
        self.num_layers = num_layers

        # Initialize embeddings for all customers and articles. Customers will have indices from 0...customers-1,
        # articles will have indices from articles...articles-1
        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim
        )
        torch.nn.init.normal_(self.embeddings.weight, std=0.1)

        self.layers = torch.nn.ModuleList()  # LightGCN layers
        for _ in range(self.num_layers):
            self.layers.append(LightGCN())

        self.sigmoid = torch.sigmoid

    def forward(self):
        raise NotImplementedError(
            "forward() has not been implemented for the GNN class. Do not use"
        )

    def gnn_propagation(self, edge_index_mp: torch.Tensor) -> torch.Tensor:
        """
        Performs the linear embedding propagation (using the LightGCN layers) and calculates final (multi-scale) embeddings
        for each user/item, which are calculated as a weighted sum of that user/item's embeddings at each layer (from
        0 to self.num_layers). Technically, the weighted sum here is the average, which is what the LightGCN authors recommend.
        args:
          edge_index_mp: a tensor of all (undirected) edges in the graph, which is used for message passing/propagation and
              calculating the multi-scale embeddings. (In contrast to the evaluation/supervision edges, which are distinct
              from the message passing edges and will be used for calculating loss/performance metrics).
        returns:
          final multi-scale embeddings for all users/items
        """
        x = self.embeddings.weight  # layer-0 embeddings

        x_at_each_layer = [
            x
        ]  # stores embeddings from each layer. Start with layer-0 embeddings
        for i in range(self.num_layers):  # now performing the GNN propagation
            x = self.layers[i](x, edge_index_mp)
            x_at_each_layer.append(x)
        final_embs = torch.stack(x_at_each_layer, dim=0).mean(
            dim=0
        )  # take average to calculate multi-scale embeddings
        return final_embs

    def predict_scores(
        self, edge_index: torch.Tensor, embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates predicted scores for each customer/article pair in the list of edges. Uses dot product of their embeddings.
        args:
          edge_index: tensor of edges (between customers and articles) whose scores we will calculate.
          embs: node embeddings for calculating predicted scores (typically the multi-scale embeddings from gnn_propagation())
        returns:
          predicted scores for each customer/article pair in edge_index
        """
        scores = (
            embs[edge_index[0, :], :] * embs[edge_index[1, :], :]
        )  # taking dot product for each customer/article pair
        scores = scores.sum(dim=1)
        scores = self.sigmoid(scores)
        return scores

    def calc_loss(
        self, data_mp: PyGData, data_pos: PyGData, data_neg: PyGData
    ) -> torch.Tensor:
        """
        The main training step. Performs GNN propagation on message passing edges, to get multi-scale embeddings.
        Then predicts scores for each training example, and calculates Bayesian Personalized Ranking (BPR) loss.
        args:
          data_mp: tensor of edges used for message passing / calculating multi-scale embeddings
          data_pos: set of positive edges that will be used during loss calculation
          data_neg: set of negative edges that will be used during loss calculation
        returns:
          loss calculated on the positive/negative training edges
        """
        # Perform GNN propagation on message passing edges to get final embeddings
        final_embs = self.gnn_propagation(data_mp.edge_index)

        # Get edge prediction scores for all positive and negative evaluation edges
        pos_scores = self.predict_scores(data_pos.edge_index, final_embs)
        neg_scores = self.predict_scores(data_neg.edge_index, final_embs)

        # # Calculate loss (binary cross-entropy). Commenting out, but can use instead of BPR if desired.
        # all_scores = torch.cat([pos_scores, neg_scores], dim=0)
        # all_labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])], dim=0)
        # loss_fn = torch.nn.BCELoss()
        # loss = loss_fn(all_scores, all_labels)

        # Calculate loss (using variation of Bayesian Personalized Ranking loss, similar to the one used in official
        # LightGCN implementation at https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py#L202)
        loss = -torch.log(self.sigmoid(pos_scores - neg_scores)).mean()
        return loss

    def evaluation(self, data_mp: PyGData, data_pos: PyGData, k: int):
        """
        Performs evaluation on validation or test set. Calculates recall@k.
        args:
          data_mp: message passing edges to use for propagation/calculating multi-scale embeddings
          data_pos: positive edges to use for scoring metrics. Should be no overlap between these edges and data_mp's edges
          k: value of k to use for recall@k
        returns:
          dictionary mapping customer ID -> recall@k on that customer
        """
        # Run propagation on the message-passing edges to get multi-scale embeddings
        final_embs = self.gnn_propagation(data_mp.edge_index)

        # Get embeddings of all unique customers in the batch of evaluation edges
        unique_customers = torch.unique_consecutive(data_pos.edge_index[0, :])
        customer_emb = final_embs[
            unique_customers, :
        ]  # has shape [number of customers in batch, 64]

        # Get embeddings of ALL articles in dataset
        article_emb = final_embs[
            self.num_customers :, :
        ]  # has shape [total number of articles in dataset, 64]

        # All ratings for each customer in batch to each article in entire dataset (using dot product as the scoring function)
        ratings = self.sigmoid(
            torch.matmul(customer_emb, article_emb.t())
        )  # shape: [# customeres in batch, # articles in dataset]
        # where entry i,j is rating of article j for customer i
        # Calculate recall@k
        result = recall_at_k(
            ratings.cpu(),
            k,
            self.num_customers,
            data_pos.edge_index.cpu(),
            unique_customers.cpu(),
            data_mp.edge_index.cpu(),
        )
        return result


class LightGCN(MessagePassing):
    """
    A single LightGCN layer. Extends the MessagePassing class from PyTorch Geometric
    """

    def __init__(self):
        super(LightGCN, self).__init__(aggr="add")  # aggregation function is 'add'

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Specifies how to perform message passing during GNN propagation. For LightGCN, we simply pass along each
        source node's embedding to the target node, normalized by the normalization term for that node.
        args:
          x_j: node embeddings of the neighbor nodes, which will be passed to the central node (shape: [E, emb_dim])
          norm: the normalization terms we calculated in forward() and passed into propagate()
        returns:
          messages from neighboring nodes j to central node i
        """
        # Here we are just multiplying the x_j's by the normalization terms (using some broadcasting)
        return norm.view(-1, 1) * x_j

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Performs the LightGCN message passing/aggregation/update to get updated node embeddings
        args:
          x: current node embeddings (shape: [N, emb_dim])
          edge_index: message passing edges (shape: [2, E])
        returns:
          updated embeddings after this layer
        """
        # Computing node degrees for normalization term in LightGCN (see LightGCN paper for details on this normalization term)
        # These will be used during message passing, to normalize each neighbor's embedding before passing it as a message
        row, col = edge_index
        deg = degree(col)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Begin propagation. Will perform message passing and aggregation and return updated node embeddings.
        return self.propagate(edge_index, x=x, norm=norm)
