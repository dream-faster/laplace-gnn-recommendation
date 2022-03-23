from torch_geometric.data import Data, Dataset
import torch


class FashionDataset(Dataset):
    """
    Dataset object containing the Spotify supervision/evaluation edges. This will be used by the DataLoader to load
    batches of edges to calculate loss or evaluation metrics on. Here, get(idx) will return ALL outgoing edges of the graph
    corresponding to playlist "idx." This is because when calculating metrics such as recall@k, we need all of the
    playlist's positive edges in the same batch.
    """

    def __init__(self, root, edge_index, transform=None, pre_transform=None):
        self.edge_index = edge_index
        self.unique_idxs = torch.unique(
            edge_index[0, :]
        ).tolist()  # playlists will all be in row 0, b/c sorted by RandLinkSplit
        self.num_nodes = len(self.unique_idxs)
        super().__init__(root, transform, pre_transform)

    def len(self):
        return self.num_nodes

    def get(self, idx):  # returns all outgoing edges associated with playlist idx
        edge_index = self.edge_index[:, self.edge_index[0, :] == idx]
        return Data(edge_index=edge_index)
