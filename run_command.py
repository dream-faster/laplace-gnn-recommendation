import argparse
from run_pipeline import run_pipeline
from run_preprocessing import preprocess
from run_pipeline_lightgcn import train
from config import link_pred_config, lightgcn_config, only_users_and_articles_nodes


def run():
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--data-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    args = parser.parse_args()

    if args.data_size is not None:
        only_users_and_articles_nodes.data_size = args.data_size
    if args.num_epochs is not None:
        link_pred_config.epochs = args.num_epochs
        lightgcn_config.epochs = args.num_epochs

    assert args.type in [
        "preprocess",
        "lightgcn",
        "encoder",
    ], "No such pipeline type exists or none given"

    if args.type == "preprocess":
        preprocess(only_users_and_articles_nodes)
    elif args.type == "lightgcn":
        train(lightgcn_config)
    elif args.type == "encoder":
        run_pipeline(link_pred_config)


if __name__ == "__main__":
    run()
