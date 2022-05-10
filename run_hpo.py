from config import Config, link_pred_config
from run_pipeline import run_pipeline
import optuna

config = link_pred_config


def train(trial):
    num_gnn_layers = trial.suggest_int("num_gnn_layers", 1, 4)

    search_space = dict(
        epochs=10,
        k=12,
        num_gnn_layers=num_gnn_layers,
        num_linear_layers=trial.suggest_int("num_linear_layers", 1, 4),
        hidden_layer_size=trial.suggest_categorical(
            "hidden_layer_size", [32, 64, 128, 256, 512]
        ),
        encoder_layer_output_size=trial.suggest_categorical(
            "encoder_layer_output_size", [32, 64, 128, 256, 512]
        ),
        conv_agg_type=trial.suggest_categorical(
            "conv_agg_type", ["add", "mean", "max"]
        ),
        heterogeneous_prop_agg_type=trial.suggest_categorical(
            "heterogeneous_prop_agg_type", ["sum", "mean", "min", "max", "mul"]
        ),
        learning_rate=trial.suggest_categorical(
            "learning_rate", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        ),
        batch_size=trial.suggest_categorical("batch_size", [24, 32, 64, 128, 256, 512]),
        num_neighbors=trial.suggest_categorical(
            "num_neighbors", [24, 32, 64, 128, 256]
        ),
        num_neighbors_it=num_gnn_layers,
        candidate_pool_size=trial.suggest_categorical(
            "candidate_pool_size", [24, 64, 128, 256]
        ),
        positive_edges_ratio=trial.suggest_categorical(
            "positive_edges_ratio", [0.2, 0.5, 0.8, 1.0]
        ),
        negative_edges_ratio=trial.suggest_categorical(
            "negative_edges_ratio", [1, 2, 5, 10, 20]
        ),
    )
    trial_config = Config(**vars(config) | search_space)
    stats = run_pipeline(trial_config)
    return 1 - stats.precision_val


study = optuna.create_study()
study.optimize(train, n_trials=40)
print(study.best_params)
study.trials_dataframe().to_csv("output/trials.csv")
