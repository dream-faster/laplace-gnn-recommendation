import pandas as pd
import torch
import numpy as np
from torch import Tensor
from config import link_pred_config, Config
from data.data_loader import create_dataloaders
from utils.get_info import select_properties

from os import listdir
from os.path import isfile, join


url = "../input/h-and-m-personalized-fashion-recommendations/"
url = "data/original/"
model_url = "model/saved"


def load_submission(url: str):
    return pd.read_csv(url + "sample_submission.csv")


def generate_submission(predictions: list[str]):

    pass


def load_model(url: str):
    """Get the model with the largest version number"""
    files = [f for f in listdir(url) if isfile(join(url, f))]

    # Get version numbers from the file names
    version_nums = [int(filename.split("_")[1].split(".")[0]) for filename in files]

    return torch.load(url + "/" + files[np.argmax(version_nums)])


def load_dataloaders(config: Config):
    _, _, test_loader, customer_id_map, article_id_map, _ = create_dataloaders(
        config.dataloader_config
    )

    return test_loader, customer_id_map, article_id_map


def map_to_id(predictions: Tensor, customer_id_map, article_id_map):
    df = pd.DataFrame(predictions.numpy())
    for row in df.itertuples():
        row[1] = row[1].item()

    pass


def make_predictions(model, dataloader):
    predictions = []
    for batch in dataloader:
        x, edge_index_dict, edge_label_index, edge_label = select_properties(batch)
        predictions.append(model.infer(x, edge_index_dict, edge_label_index))

    return torch.concat(predictions)


def save_csv(df: pd.DataFrame):
    df.to_csv("data/derived/submission.csv", index=False)


def submission_pipeline(config: Config):
    # sample = load_submission(url)

    model = load_model(model_url)
    dataloader, customer_id_map, article_id_map = load_dataloaders(config)

    predictions = make_predictions(model, dataloader)
    prediction_user_id = map_to_id(predictions, customer_id_map, article_id_map)
    csv = generate_submission(prediction_user_id)
    save_csv(csv)


if __name__ == "__main__":
    submission_pipeline(link_pred_config)
