import csv

from src.tune import build_summary_table, main, run_trial, sample_configs


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


def test_build_summary_table_reports_range_and_optimal_from_best_trial():
    search_space = {"batch_size": [16, 32], "lr": [1e-2, 1e-3]}
    configs = [
        {"batch_size": 16, "lr": 1e-2},
        {"batch_size": 32, "lr": 1e-3},
    ]
    results = [
        {"val_loss": 0.9, "val_acc": 0.7},
        {"val_loss": 0.3, "val_acc": 0.9},  # best (lowest val_loss)
    ]

    table = build_summary_table(search_space, configs, results)

    rows_by_name = {row["hyperparameter"]: row for row in table}
    assert rows_by_name["batch_size"]["range"] == [16, 32]
    assert rows_by_name["batch_size"]["optimal"] == 32
    assert rows_by_name["lr"]["optimal"] == 1e-3


def test_main_runs_search_and_writes_results_and_summary(tmp_path):
    train_manifest = tmp_path / "train.csv"
    val_manifest = tmp_path / "val.csv"
    _write_manifest(train_manifest)
    _write_manifest(val_manifest)

    out_dir = tmp_path / "search"

    main([
        "--model", "cnn",
        "--mode", "scratch",
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--n-trials", "2",
        "--epochs-per-trial", "1",
        "--seed", "42",
        "--out-dir", str(out_dir),
    ])

    with open(out_dir / "results.csv") as f:
        results_rows = list(csv.DictReader(f))
    assert len(results_rows) == 2
    assert "val_loss" in results_rows[0]

    with open(out_dir / "summary.csv") as f:
        summary_rows = list(csv.DictReader(f))
    assert {"hyperparameter", "range", "optimal"} <= set(summary_rows[0].keys())
