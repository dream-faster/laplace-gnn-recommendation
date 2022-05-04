import argparse
from run_pipeline import run_pipeline
from run_preprocessing import preprocess
from run_pipeline_lightgcn import train
from config import link_pred_config, lightgcn_config, only_users_and_articles_nodes


def run():
    parser = argparse.ArgumentParser()

    # Add one special argument for deciding what pipeline to run
    parser.add_argument("--type", type=str, default=None)

    # Go through each item in configs and add it to the parser
    for key, value in vars(link_pred_config).items():
        parser.add_argument(f"--{key.replace('_','-')}", type=type(value), default=None)
    for key, value in vars(only_users_and_articles_nodes).items():
        parser.add_argument(f"--{key.replace('_','-')}", type=type(value), default=None)

    args = parser.parse_args()

    # Decide which pipeline to configure
    if vars(args)["type"] == "preprocess":
        config = only_users_and_articles_nodes
    elif vars(args)["type"] == "lightgcn":
        config = lightgcn_config
    else:
        config = link_pred_config

    # Overwrite defaults in config objects with arguments from parser
    for key, value in vars(args).items():
        if value is not None:
            vars(config)[key] = value

    assert args.type in [
        "preprocess",
        "lightgcn",
        "encoder",
    ], "No such pipeline type exists or none given"

    # Run set pipeline
    if args.type == "preprocess":
        preprocess(config)
    elif args.type == "lightgcn":
        train(config)
    elif args.type == "encoder":
        run_pipeline(config)


if __name__ == "__main__":
    run()
