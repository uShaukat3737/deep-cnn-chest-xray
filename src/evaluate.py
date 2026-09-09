import argparse
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import ChestXrayDataset, build_transforms
from src.models.cnn import ChestCNN
from src.models.pretrained import build_model as build_pretrained_model

NUM_CLASSES = 2


@torch.no_grad()
def predict(model, loader, device):
    model.to(device)
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())
    return y_true, y_pred, y_prob


def _confusion_matrix(y_true, y_pred):
    matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def _precision_recall_f1(matrix, cls):
    tp = matrix[cls][cls]
    fp = sum(matrix[r][cls] for r in range(NUM_CLASSES)) - tp
    fn = sum(matrix[cls][c] for c in range(NUM_CLASSES)) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_auc_roc(y_true, y_scores):
    # Mann-Whitney U statistic == AUC-ROC; average ranks handle score ties.
    order = sorted(range(len(y_scores)), key=lambda i: y_scores[i])
    ranks = [0.0] * len(y_scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and y_scores[order[j]] == y_scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    n_pos = sum(1 for t in y_true if t == 1)
    n_neg = len(y_true) - n_pos
    rank_sum_pos = sum(ranks[i] for i in range(len(y_true)) if y_true[i] == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_metrics(y_true, y_pred):
    matrix = _confusion_matrix(y_true, y_pred)
    accuracy = sum(matrix[i][i] for i in range(NUM_CLASSES)) / len(y_true)
    per_class = [_precision_recall_f1(matrix, c) for c in range(NUM_CLASSES)]
    macro = {
        key: sum(pc[key] for pc in per_class) / NUM_CLASSES
        for key in ("precision", "recall", "f1")
    }
    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro": macro,
    }


def _plot_confusion_matrix(matrix, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_roc_curve(y_true, y_prob, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    thresholds = sorted(set(y_prob), reverse=True)
    n_pos = sum(1 for t in y_true if t == 1)
    n_neg = len(y_true) - n_pos
    tpr, fpr = [0.0], [0.0]
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in y_prob]
        tp = sum(1 for t, p in zip(y_true, preds) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, preds) if t == 0 and p == 1)
        tpr.append(tp / n_pos if n_pos else 0.0)
        fpr.append(fp / n_neg if n_neg else 0.0)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.savefig(out_path)
    plt.close(fig)


def _build_model(name, mode):
    if name == "cnn":
        return ChestCNN(num_classes=NUM_CLASSES)
    return build_pretrained_model(name, mode=mode, num_classes=NUM_CLASSES)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "resnet50", "mobilenetv2"])
    parser.add_argument("--mode", required=True, choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--comparison-csv", required=True)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(args.model, args.mode)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    _, val_tf = build_transforms()
    test_ds = ChestXrayDataset(args.test_manifest, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=32)

    y_true, y_pred, y_prob = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    auc = compute_auc_roc(y_true, y_prob)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    _plot_roc_curve(y_true, y_prob, out_dir / "roc_curve.png")

    row = {
        "model": args.model,
        "mode": args.mode,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro"]["precision"],
        "macro_recall": metrics["macro"]["recall"],
        "macro_f1": metrics["macro"]["f1"],
        "auc_roc": auc,
    }
    fieldnames = list(row.keys())
    write_header = not os.path.exists(args.comparison_csv)
    with open(args.comparison_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
