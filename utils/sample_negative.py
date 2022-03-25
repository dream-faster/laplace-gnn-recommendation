import torch
from torch_geometric.data import Data as PyGData


def sample_negative_edges(
    batch: PyGData, num_customers: int, num_nodes: int
) -> PyGData:
    # Randomly samples articles for each customer. Doesn't currently check if they are true negatives, since that is
    # computationally expensive. This is fine in our case, because we will never be sampling more than ~100
    # articles for a user (out of thousands of articles), so although we will accidentally sample some positive articles,
    # it will be an acceptably small number. However, if that is not the case for your dataset, please consider
    # sampling true negatives only. Here we sample 1 negative edge for each positive edge in the graph, so we will
    # end up having a balanced 1:1 ratio of positive to negative edges.
    negs = []
    for i in batch.edge_index[0, :]:  # looping over playlists
        assert i < num_customers  # just ensuring that i is a customer
        rand = torch.randint(num_customers, num_nodes, (1,))  # randomly sample a song
        negs.append(rand.item())
    edge_index_negs = torch.row_stack([batch.edge_index[0, :], torch.LongTensor(negs)])
    return PyGData(edge_index=edge_index_negs)
