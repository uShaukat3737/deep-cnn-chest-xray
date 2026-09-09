import csv
import itertools
import random

from src.train import main as train_main

SEARCH_SPACE = {
    "batch_size": [16, 32, 64],
    "lr": [1e-2, 1e-3, 1e-4],
    "dropout": [0.2, 0.5],
    "patience": [3, 5],
    "l1": [0.0, 1e-5],
    "l2": [0.0, 1e-4, 1e-3],
    "norm_scheme": ["imagenet", "dataset_stats"],
    "augmentation": ["on", "off"],
}


def sample_configs(search_space, n_trials, seed):
    keys = list(search_space.keys())
    full_grid = list(itertools.product(*(search_space[k] for k in keys)))
    rng = random.Random(seed)
    sampled = rng.sample(full_grid, min(n_trials, len(full_grid)))
    return [dict(zip(keys, combo)) for combo in sampled]


def run_trial(config, model, mode, train_manifest, val_manifest, epochs, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = work_dir / "checkpoints"
    log_path = work_dir / "log.csv"

    train_main([
        "--model", model,
        "--mode", mode,
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--epochs", str(epochs),
        "--batch-size", str(config.get("batch_size", 32)),
        "--lr", str(config.get("lr", 1e-3)),
        "--dropout", str(config.get("dropout", 0.5)),
        "--patience", str(config.get("patience", 5)),
        "--weight-decay", str(config.get("l2", 0.0)),
        "--l1", str(config.get("l1", 0.0)),
        "--norm-scheme", config.get("norm_scheme", "imagenet"),
        "--augmentation", config.get("augmentation", "on"),
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-path", str(log_path),
    ])

    with open(log_path) as f:
        last_row = list(csv.DictReader(f))[-1]
    return {"val_loss": float(last_row["val_loss"]), "val_acc": float(last_row["val_acc"])}
