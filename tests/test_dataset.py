import torch

from src.dataset import build_transforms


def test_train_transform_is_stochastic_val_transform_is_deterministic():
    from PIL import Image

    img = Image.new("RGB", (300, 300), "white")
    train_tf, val_tf = build_transforms()

    torch.manual_seed(0)
    out1 = train_tf(img)
    torch.manual_seed(1)
    out2 = train_tf(img)
    assert not torch.equal(out1, out2)

    val_out1 = val_tf(img)
    val_out2 = val_tf(img)
    assert torch.equal(val_out1, val_out2)
    assert val_out1.shape == (3, 224, 224)
