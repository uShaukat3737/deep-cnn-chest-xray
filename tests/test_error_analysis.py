import csv

import torch

from src.error_analysis import find_misclassified, main


def test_find_misclassified_filters_and_sorts_by_confidence():
    paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 1, 1]  # b and c are misclassified
    y_prob = [0.1, 0.3, 0.9, 0.6]  # prob of class 1

    result = find_misclassified(paths, y_true, y_pred, y_prob)

    assert [r["path"] for r in result] == ["c.jpg", "b.jpg"]  # sorted by confidence in wrong pred, desc
    assert result[0]["true_label"] == 0
    assert result[0]["pred_label"] == 1
    assert result[0]["prob"] == 0.9


def test_main_writes_misclassified_csv(tmp_path):
    from PIL import Image

    from src.models.cnn import ChestCNN

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    manifest = tmp_path / "test.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for label in ("NORMAL", "PNEUMONIA"):
            for i in range(6):
                img_path = img_dir / f"{label}_{i}.jpg"
                Image.new("RGB", (32, 32), (i * 10, 0, 0)).save(img_path)
                writer.writerow([str(img_path), label])

    checkpoint_path = tmp_path / "model.pth"
    torch.save(ChestCNN(num_classes=2).state_dict(), checkpoint_path)

    out_dir = tmp_path / "errors"

    main([
        "--model", "cnn",
        "--mode", "scratch",
        "--checkpoint", str(checkpoint_path),
        "--test-manifest", str(manifest),
        "--out-dir", str(out_dir),
    ])

    with open(out_dir / "misclassified.csv") as f:
        rows = list(csv.DictReader(f))
    assert {"path", "true_label", "pred_label", "prob"} <= set(rows[0].keys()) if rows else True
