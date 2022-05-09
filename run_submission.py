import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from config import link_pred_config, Config
from data.data_loader import create_dataloaders
from utils.get_info import select_properties

from os import listdir
from os.path import isfile, join


def load_model(url: str):
    """Get the model with the largest version number"""
    files = [f for f in listdir(url) if isfile(join(url, f))]

    # Get version numbers from the file names
    version_nums = [int(filename.split("_")[1].split(".")[0]) for filename in files]

    return torch.load(url + "/" + files[np.argmax(version_nums)])


def load_dataloaders(config: Config):
    _, _, test_loader, customer_id_map, article_id_map, _ = create_dataloaders(config)

    return test_loader, customer_id_map, article_id_map


def map_to_id(
    predictions: Tensor, customer_id_map: dict, article_id_map: dict
) -> pd.DataFrame:
    df = pd.DataFrame(predictions.numpy())

    for col in df.columns:
        df[col] = df[col].map(str).map(article_id_map)

    df["customer_id"] = df.index.to_series().map(lambda x: customer_id_map[str(x)])
    df["prediction"] = (
        df[[column for column in df.columns if column != "customer_id"]]
        .astype(str)
        .agg(" ".join, axis=1)
    )

    return df


@torch.no_grad()
def make_predictions(model, dataloader, k: int) -> Tensor:
    predictions = []

    for batch in tqdm(dataloader):
        # Prepare data
        x, edge_index_dict, edge_label_index, edge_label = select_properties(batch)

        # Get predictions
        prediction = model.infer(x, edge_index_dict, edge_label_index)

        # Filter out positive edges from prediction and edge_label_index
        prediction = prediction[edge_label == 0]
        filtered_edge_label_index = edge_label_index[:, edge_label == 0]

        # Get top k predictions and indecies from only negative edges
        _, topK_indecies = torch.topk(prediction, k=k)
        top_articles = filtered_edge_label_index[1][topK_indecies]

        predictions.append(top_articles)

    return torch.stack(predictions)


def save_csv(df: pd.DataFrame):
    """Save selected columns to csv"""
    cols_to_keep = ["customer_id", "prediction"]
    df.loc[:, cols_to_keep].to_csv("data/derived/submission.csv", index=False)


def submission_pipeline(config: Config):
    print("| Loading Model...")
    model = load_model("model/saved")

    print("| Loading Dataloaders...")
    dataloader, customer_id_map, article_id_map = load_dataloaders(config)

    print("| Making Predictions...")
    predictions = make_predictions(model, dataloader, k=config.k)

    print("| Mapping positions to ids...")
    prediction_mapped = map_to_id(predictions, customer_id_map, article_id_map)

    print("| Saving predictions...")
    save_csv(prediction_mapped)


if __name__ == "__main__":
    submission_pipeline(link_pred_config)
