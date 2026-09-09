from src.train import EarlyStopping


def test_early_stopping_triggers_after_patience_non_improving_epochs():
    stopper = EarlyStopping(patience=3)

    losses = [1.0, 0.9, 0.95, 0.96, 0.97, 0.98]
    stopped_at = None
    for epoch, loss in enumerate(losses):
        if stopper.step(loss):
            stopped_at = epoch
            break

    assert stopped_at == 4  # 3 non-improving epochs after the best (epoch 1)
