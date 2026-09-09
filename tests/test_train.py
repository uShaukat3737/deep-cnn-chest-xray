import torch.nn as nn

from src.train import Checkpointer, EarlyStopping


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
