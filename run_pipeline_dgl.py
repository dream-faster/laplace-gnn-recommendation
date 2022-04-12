import dgl
import torch
import numpy as np


graph = torch.load("data/derived/graph.pt")

device = "cpu"  # change to 'cuda' for GPU

negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)

sampler = dgl.dataloading.NeighborSampler([4, 4])
sampler = dgl.dataloading.as_edge_prediction_sampler(
    sampler, negative_sampler=negative_sampler
)

train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DataLoader.
    graph,  # The graph
    torch.arange(graph.number_of_edges()),  # The edges to iterate over
    sampler,  # The neighbor sampler
    device=device,  # Put the MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=1024,  # Batch size
    shuffle=True,  # Whether to shuffle the nodes for every epoch
    drop_last=False,  # Whether to drop the last incomplete batch
    num_workers=0,  # Number of sampler processes
)

# dataloader = dgl.dataloading.EdgeDataLoader(
#     g,
#     train_seeds,
#     sampler,
#     negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
#     batch_size=args.batch_size,
#     shuffle=True,
#     drop_last=False,
#     pin_memory=True,
#     num_workers=args.num_workers,
# )
