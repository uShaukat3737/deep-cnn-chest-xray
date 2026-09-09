import argparse
import csv
from pathlib import Path

from src.data_prep import subsample_by_fraction
from src.train import main as train_main


def _read_manifest(path):
    with open(path, newline="") as f:
        return [(row["path"], row["label"]) for row in csv.DictReader(f)]


def _write_manifest(path, items):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(items)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "resnet50", "mobilenetv2"])
    parser.add_argument("--mode", required=True, choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_train_items = _read_manifest(args.train_manifest)
    fractions = [float(f) for f in args.fractions.split(",")]

    rows = []
    for fraction in fractions:
        subset = subsample_by_fraction(full_train_items, fraction=fraction, seed=args.seed)
        work_dir = out_dir / f"frac_{fraction}"
        work_dir.mkdir(parents=True, exist_ok=True)
        subset_manifest = work_dir / "train.csv"
        _write_manifest(subset_manifest, subset)

        checkpoint_dir = work_dir / "checkpoints"
        log_path = work_dir / "log.csv"
        train_main([
            "--model", args.model,
            "--mode", args.mode,
            "--train-manifest", str(subset_manifest),
            "--val-manifest", args.val_manifest,
            "--epochs", str(args.epochs),
            "--checkpoint-dir", str(checkpoint_dir),
            "--log-path", str(log_path),
        ])

        with open(log_path) as f:
            last_row = list(csv.DictReader(f))[-1]
        rows.append({
            "fraction": fraction,
            "n_train_samples": len(subset),
            "val_loss": last_row["val_loss"],
            "val_acc": last_row["val_acc"],
        })

    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fraction", "n_train_samples", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
