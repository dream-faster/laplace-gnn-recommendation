import torch
from torch import Tensor
from data.types import DataLoaderConfig, FeatureInfo
from torch.nn import Linear, Embedding, ModuleList
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class Encoder_Decoder_Model_Homo(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encoder(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decoder(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
