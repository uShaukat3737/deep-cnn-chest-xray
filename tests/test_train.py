import torch
import torch.nn as nn

import csv

from src.train import Checkpointer, EarlyStopping, main, train_one_epoch


def test_early_stopping_triggers_after_patience_non_improving_epochs():
    stopper = EarlyStopping(patience=3)

    losses = [1.0, 0.9, 0.95, 0.96, 0.97, 0.98]
    stopped_at = None
    for epoch, loss in enumerate(losses):
        if stopper.step(loss):
            stopped_at = epoch
            break

    assert stopped_at == 4  # 3 non-improving epochs after the best (epoch 1)


def test_checkpointer_saves_only_on_improvement(tmp_path):
    model = nn.Linear(2, 2)
    ckpt_path = tmp_path / "best.pth"
    checkpointer = Checkpointer(ckpt_path)

    saved_1 = checkpointer.step(model, val_loss=1.0)
    mtime_1 = ckpt_path.stat().st_mtime_ns

    saved_2 = checkpointer.step(model, val_loss=1.5)  # worse, should not overwrite

    saved_3 = checkpointer.step(model, val_loss=0.5)  # better, should overwrite
    mtime_3 = ckpt_path.stat().st_mtime_ns

    assert saved_1 is True
    assert saved_2 is False
    assert saved_3 is True
    assert mtime_3 >= mtime_1


def test_train_one_epoch_reduces_loss_on_synthetic_batch():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = [(x, y)] * 5
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        loss_before = criterion(model(x), y).item()

    avg_loss = train_one_epoch(model, loader, optimizer, criterion, device="cpu")

    with torch.no_grad():
        loss_after = criterion(model(x), y).item()

    assert isinstance(avg_loss, float)
    assert loss_after < loss_before


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


def test_main_trains_cnn_and_writes_checkpoint_and_log(tmp_path):
    train_manifest = tmp_path / "train.csv"
    val_manifest = tmp_path / "val.csv"
    _write_manifest(train_manifest)
    _write_manifest(val_manifest)

    checkpoint_dir = tmp_path / "checkpoints"
    log_path = tmp_path / "log.csv"

    main([
        "--model", "cnn",
        "--mode", "scratch",
        "--train-manifest", str(train_manifest),
        "--val-manifest", str(val_manifest),
        "--epochs", "1",
        "--batch-size", "4",
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-path", str(log_path),
    ])

    assert (checkpoint_dir / "cnn_scratch_best.pth").exists()
    with open(log_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["epoch", "train_loss", "val_loss", "val_acc"]
    assert len(rows) == 2
