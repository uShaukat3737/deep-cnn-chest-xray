import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import ChestXrayDataset, build_transforms
from src.evaluate import _build_model, predict


def find_misclassified(paths, y_true, y_pred, y_prob):
    misclassified = [
        {"path": path, "true_label": t, "pred_label": p, "prob": prob}
        for path, t, p, prob in zip(paths, y_true, y_pred, y_prob)
        if t != p
    ]
    misclassified.sort(key=lambda r: abs(r["prob"] - 0.5), reverse=True)
    return misclassified


def _plot_grid(records, out_path):
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return
    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, records):
        ax.imshow(Image.open(r["path"]))
        ax.set_title(f"true={r['true_label']} pred={r['pred_label']} p={r['prob']:.2f}", fontsize=8)
        ax.axis("off")
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "resnet50", "mobilenetv2"])
    parser.add_argument("--mode", required=True, choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(args.model, args.mode)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    _, val_tf = build_transforms()
    test_ds = ChestXrayDataset(args.test_manifest, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=32)
    paths = [path for path, _ in test_ds.items]

    print(f"running inference on {len(test_ds)} test samples", flush=True)
    y_true, y_pred, y_prob = predict(model, test_loader, device)
    misclassified = find_misclassified(paths, y_true, y_pred, y_prob)
    print(f"found {len(misclassified)} misclassified out of {len(y_true)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "misclassified.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "pred_label", "prob"])
        writer.writeheader()
        writer.writerows(misclassified)

    top = misclassified[:args.top_n]
    if top:
        _plot_grid(top, out_dir / "misclassified_grid.png")


if __name__ == "__main__":
    main()
