from torch import nn
from copy import deepcopy
from torch.nn import Linear, LayerNorm, BatchNorm2d


def get_SAGEConv_layers(
    num_layers: int, hidden_channels: int, out_channels: int
) -> nn.ModuleList:
    from torch_geometric.nn import SAGEConv

    conv_single_layer = SAGEConv(
        (-1, -1),
        hidden_channels,
        aggr="add",
        normalize=False,
        bias=True,
    )
    conv_last_layer = SAGEConv(
        (-1, -1),
        out_channels,
        aggr="add",
        normalize=False,
        bias=True,
    )

    if num_layers == 1:
        return nn.ModuleList([conv_last_layer])
    else:
        return nn.ModuleList(
            [deepcopy(conv_single_layer) for _ in range(num_layers - 1)]
            + [conv_last_layer]
        )


def get_linear_layers(
    num_layers: int, in_channels: int, hidden_channels: int, out_channels: int
) -> nn.ModuleList:
    first_layer = Linear(in_channels, hidden_channels)
    middle_layers = Linear(hidden_channels, hidden_channels)
    last_layer = Linear(hidden_channels, out_channels)

    if num_layers == 1:
        return nn.ModuleList([Linear(in_channels, out_channels)])
    if num_layers == 2:
        return nn.ModuleList(
            [
                Linear(in_channels, hidden_channels),
                Linear(hidden_channels, out_channels),
            ]
        )

    return nn.ModuleList(
        [first_layer]
        + [deepcopy(middle_layers) for _ in range(num_layers - 2)]
        + [last_layer]
    )
