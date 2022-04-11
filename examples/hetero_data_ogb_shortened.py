from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T

hetero_data = OGB_MAG("data/ogb")[0]

# hetero_data = T.ToUndirected()(hetero_data)
# del hetero_data["institution", "rev_affiliated_with", "author"].edge_label
# del hetero_data["paper", "rev_writes", "author"].edge_label
# del hetero_data["paper", "rev_cites", "paper"].edge_label
# del hetero_data["field_of_study", "rev_has_topic", "paper"].edge_label

transform = RandomLinkSplit(
    is_undirected=False,
    add_negative_train_samples=False,
    num_val=0.01,
    num_test=0.01,
    neg_sampling_ratio=0,
    edge_types=[
        ("author", "affiliated_with", "institution"),
        ("author", "writes", "paper"),
        ("paper", "cites", "paper"),
        ("paper", "has_topic", "field_of_study"),
    ],
    rev_edge_types=[
        ("institution", "rev_affiliated_with", "author"),
        ("paper", "rev_writes", "author"),
        ("paper", "rev_cites", "paper"),
        ("field_of_study", "rev_has_topic", "paper"),
    ],
)

train_split, val_split, test_split = transform(hetero_data)

loader = NeighborLoader(
    train_split,
    # Sample 30 neighbors for each node and edge type for 2 iterations
    num_neighbors=[5],  # {key: [30] * 2 for key in train_split.edge_types},
    # Use a batch size of 128 for sampling training nodes of type paper
    batch_size=128,
    input_nodes=("paper", None),
)

sampled_hetero_data = next(iter(loader))
print(sampled_hetero_data["paper"].batch_size)
