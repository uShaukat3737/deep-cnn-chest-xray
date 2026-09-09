import argparse
import csv
import itertools
import random
from pathlib import Path

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


def build_summary_table(search_space, configs, results):
    best_idx = min(range(len(results)), key=lambda i: results[i]["val_loss"])
    best_config = configs[best_idx]
    return [
        {"hyperparameter": name, "range": values, "optimal": best_config[name]}
        for name, values in search_space.items()
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "resnet50", "mobilenetv2"])
    parser.add_argument("--mode", required=True, choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs-per-trial", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = sample_configs(SEARCH_SPACE, n_trials=args.n_trials, seed=args.seed)
    results = []
    for i, config in enumerate(configs):
        print(f"[trial {i + 1}/{len(configs)}] config={config}", flush=True)
        work_dir = out_dir / f"trial_{i}"
        result = run_trial(
            config, model=args.model, mode=args.mode,
            train_manifest=args.train_manifest, val_manifest=args.val_manifest,
            epochs=args.epochs_per_trial, work_dir=work_dir,
        )
        results.append(result)
        print(f"[trial {i + 1}/{len(configs)}] val_loss={result['val_loss']:.4f} val_acc={result['val_acc']:.4f}", flush=True)

    with open(out_dir / "results.csv", "w", newline="") as f:
        fieldnames = list(SEARCH_SPACE.keys()) + ["val_loss", "val_acc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for config, result in zip(configs, results):
            writer.writerow({**config, **result})

    summary = build_summary_table(SEARCH_SPACE, configs, results)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hyperparameter", "range", "optimal"])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
