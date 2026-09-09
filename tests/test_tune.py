import csv

from src.tune import run_trial, sample_configs


def test_sample_configs_returns_unique_deterministic_configs():
    search_space = {
        "batch_size": [16, 32, 64],
        "lr": [1e-2, 1e-3, 1e-4],
        "dropout": [0.2, 0.5],
    }

    configs_a = sample_configs(search_space, n_trials=5, seed=42)
    configs_b = sample_configs(search_space, n_trials=5, seed=42)

    assert configs_a == configs_b
    assert len(configs_a) == 5

    seen = {tuple(sorted(c.items())) for c in configs_a}
    assert len(seen) == 5

    for config in configs_a:
        assert set(config.keys()) == {"batch_size", "lr", "dropout"}
        assert config["batch_size"] in search_space["batch_size"]


def _write_manifest(path, n_per_class=4):
    from PIL import Image

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


def test_run_trial_returns_metrics_and_writes_checkpoint(tmp_path):
    train_manifest = tmp_path / "train.csv"
    val_manifest = tmp_path / "val.csv"
    _write_manifest(train_manifest)
    _write_manifest(val_manifest)

    work_dir = tmp_path / "trial_0"
    config = {"batch_size": 4, "lr": 1e-3, "dropout": 0.5, "patience": 5}

    result = run_trial(
        config, model="cnn", mode="scratch",
        train_manifest=str(train_manifest), val_manifest=str(val_manifest),
        epochs=1, work_dir=work_dir,
    )

    assert "val_loss" in result and "val_acc" in result
    assert isinstance(result["val_loss"], float)
    assert (work_dir / "checkpoints" / "cnn_scratch_best.pth").exists()
