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
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--positive-edges-ratio", type=int, default=None)
    parser.add_argument("--negative-edges-ratio", type=int, default=None)
    args = parser.parse_args()

    if args.data_size is not None:
        only_users_and_articles_nodes.data_size = args.data_size
    if args.num_epochs is not None:
        link_pred_config.epochs = args.num_epochs
        lightgcn_config.epochs = args.num_epochs
    if args.num_layers is not None:
        link_pred_config.num_layers = args.num_layers
        lightgcn_config.num_layers = args.num_layers
    if args.candidate_pool_size is not None:
        link_pred_config.candidate_pool_size = args.candidate_pool_size
        lightgcn_config.candidate_pool_size = args.candidate_pool_size
    if args.positive_edges_ratio is not None:
        link_pred_config.positive_edges_ratio = args.positive_edges_ratio
        lightgcn_config.positive_edges_ratio = args.positive_edges_ratio
    if args.negative_edges_ratio is not None:
        link_pred_config.negative_edges_ratio = args.negative_edges_ratio
        lightgcn_config.negative_edges_ratio = args.negative_edges_ratio

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
