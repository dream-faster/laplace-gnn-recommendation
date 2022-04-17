import torch_geometric
from data.data_loader import create_dataloaders
from config import config, Config
from utils.visualize import visualize_graph

(
    train_loader,
    val_loader,
    test_loader,
    customer_id_map,
    article_id_map,
    full_data,
) = create_dataloaders(config.dataloader_config)


visualize_graph(next(iter(train_loader)))
