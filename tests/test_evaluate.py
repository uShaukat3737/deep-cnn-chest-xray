import pytest
import torch
import torch.nn as nn

import csv

from src.evaluate import compute_auc_roc, compute_metrics, main, predict


def test_compute_metrics_matches_hand_computed_confusion_case():
    # 4 NORMAL (0), 4 PNEUMONIA (1); 1 false negative, 1 false positive
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]

    metrics = compute_metrics(y_true, y_pred)

    # class 0: TP=3, FP=1, FN=1 -> precision=0.75, recall=0.75, f1=0.75
    # class 1: TP=3, FP=1, FN=1 -> precision=0.75, recall=0.75, f1=0.75
    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["confusion_matrix"] == [[3, 1], [1, 3]]
    assert metrics["per_class"][0]["precision"] == pytest.approx(0.75)
    assert metrics["per_class"][0]["recall"] == pytest.approx(0.75)
    assert metrics["per_class"][0]["f1"] == pytest.approx(0.75)
    assert metrics["macro"]["precision"] == pytest.approx(0.75)
    assert metrics["macro"]["recall"] == pytest.approx(0.75)
    assert metrics["macro"]["f1"] == pytest.approx(0.75)


def test_compute_auc_roc_perfect_separation_gives_one():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]

    auc = compute_auc_roc(y_true, y_scores)

    assert auc == pytest.approx(1.0)


def test_compute_auc_roc_random_guessing_gives_half():
    y_true = [0, 1, 0, 1]
    y_scores = [0.5, 0.5, 0.5, 0.5]

    auc = compute_auc_roc(y_true, y_scores)

    assert auc == pytest.approx(0.5)


def test_predict_returns_true_pred_and_prob_lists():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x1, y1 = torch.randn(3, 4), torch.tensor([0, 1, 0])
    x2, y2 = torch.randn(2, 4), torch.tensor([1, 0])
    loader = [(x1, y1), (x2, y2)]

    y_true, y_pred, y_prob = predict(model, loader, device="cpu")

    assert y_true == [0, 1, 0, 1, 0]
    assert len(y_pred) == 5
    assert all(0.0 <= p <= 1.0 for p in y_prob)


def test_main_evaluates_checkpoint_and_appends_comparison_row(tmp_path):
    from PIL import Image

    from src.models.cnn import ChestCNN

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    manifest = tmp_path / "test.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for label in ("NORMAL", "PNEUMONIA"):
            for i in range(3):
                img_path = img_dir / f"{label}_{i}.jpg"
                Image.new("RGB", (32, 32), (i * 10, 0, 0)).save(img_path)
                writer.writerow([str(img_path), label])

    checkpoint_path = tmp_path / "model.pth"
    torch.save(ChestCNN(num_classes=2).state_dict(), checkpoint_path)

    out_dir = tmp_path / "eval_out"
    comparison_csv = tmp_path / "comparison.csv"

    main([
        "--model", "cnn",
        "--mode", "scratch",
        "--checkpoint", str(checkpoint_path),
        "--test-manifest", str(manifest),
        "--out-dir", str(out_dir),
        "--comparison-csv", str(comparison_csv),
    ])

    with open(comparison_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model"] == "cnn"
    assert {"accuracy", "macro_precision", "macro_recall", "macro_f1", "auc_roc"} <= set(rows[0].keys())
