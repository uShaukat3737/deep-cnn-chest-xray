import csv

from PIL import Image

from src.data_size_study import main


def _write_manifest(path, n_per_class=8):
    img_dir = path.parent / "imgs"
    img_dir.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for label in ("NORMAL", "PNEUMONIA"):
            for i in range(n_per_class):
                img_path = img_dir / f"{label}_{i}.jpg"
                Image.new("RGB", (32, 32), (i * 10, 0, 0)).save(img_path)
                writer.writerow([str(img_path), label])


def test_main_trains_at_each_fraction_and_writes_results(tmp_path):
    train_manifest = tmp_path / "train.csv"
    val_manifest = tmp_path / "val.csv"
    _write_manifest(train_manifest)
    _write_manifest(val_manifest)

    out_dir = tmp_path / "size_study"

    main([
        "--model", "cnn",
        "--mode", "scratch",
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--fractions", "0.5,1.0",
        "--epochs", "1",
        "--seed", "42",
        "--out-dir", str(out_dir),
    ])

    with open(out_dir / "results.csv") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert {"fraction", "n_train_samples", "val_loss", "val_acc"} <= set(rows[0].keys())
    fractions = {float(r["fraction"]) for r in rows}
    assert fractions == {0.5, 1.0}
