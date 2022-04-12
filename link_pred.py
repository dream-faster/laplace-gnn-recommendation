from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
from hm_dataset import HMDataset
from torch_geometric.nn import SAGEConv, to_hetero


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = HMDataset("./data")
data = dataset[0].to(device)
data["article"].x = data["article"].x.float()
data["customer"].x = data["customer"].x.float()
data[("customer", "buys", "article")].edge_index = data[
    ("customer", "buys", "article")
].edge_index.long()

# Add a reverse ('article', 'rev_buys', 'customer') relation for message passing:
data = T.ToUndirected()(data)
# del data["article", "rev_buys", "customer"].edge_label  # Remove "reverse" label.

# Perform a link-level split into training, validation, and test edges:
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[("customer", "buys", "article")],
    rev_edge_types=[("article", "rev_buys", "customer")],
    is_undirected=True,
)(data)


train_loader = LinkNeighborLoader(
    train_data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[8],
    # Use a batch size of 128 for sampling training nodes
    batch_size=8,
    edge_label_index=("customer", "buys", "article"),
    directed=False,
    replace=True,
    shuffle=True,
)
val_loader = LinkNeighborLoader(
    val_data,
    num_neighbors=[8],
    batch_size=8,
    edge_label_index=("customer", "buys", "article"),
    directed=False,
    replace=True,
    shuffle=True,
)
test_loader = LinkNeighborLoader(
    test_data,
    num_neighbors=[8],
    batch_size=8,
    edge_label_index=("customer", "buys", "article"),
    directed=False,
    replace=True,
    shuffle=True,
)


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict["customer"][row], z_dict["article"][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)

# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:
with torch.no_grad():
    batch = next(iter(train_loader))
    model.encoder(batch.x_dict, batch.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(train_data):
    model.train()
    optimizer.zero_grad()
    pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data["customer", "article"].edge_label_index,
    )
    target = train_data["customer", "article"].edge_label
    # TODO figure out loss
    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(
        data.x_dict,
        data.edge_index_dict,
        data["customer", "article"].edge_label_index,
    )
    pred = pred.clamp(min=0, max=5)
    target = data["customer", "article"].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


for epoch in range(1, 301):
    for batch in tqdm(train_loader):
        loss = train(batch)
        train_rmse = test(batch)
    for batch in tqdm(val_loader):
        val_rmse = test(val_data)
    for batch in tqdm(test_loader):
        test_rmse = test(batch)
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, "
        f"Val: {val_rmse:.4f}, Test: {test_rmse:.4f}"
    )
