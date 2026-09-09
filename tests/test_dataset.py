import csv

import torch

from src.dataset import ChestXrayDataset, build_sample_weights, build_transforms


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


def test_chest_xray_dataset_returns_tensor_and_int_label(tmp_path):
    from PIL import Image

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (50, 50), "white").save(img_path)

    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerow([str(img_path), "NORMAL"])
        writer.writerow([str(img_path), "PNEUMONIA"])

    _, val_tf = build_transforms()
    ds = ChestXrayDataset(manifest, transform=val_tf)

    assert len(ds) == 2
    img_tensor, label = ds[0]
    assert img_tensor.shape == (3, 224, 224)
    assert label == 0

    _, label2 = ds[1]
    assert label2 == 1


def test_build_sample_weights_gives_minority_class_higher_weight():
    labels = [0, 0, 0, 1]  # 3 NORMAL, 1 PNEUMONIA
    weights = build_sample_weights(labels)

    assert len(weights) == 4
    assert weights[3] > weights[0]
    assert weights[0] == weights[1] == weights[2]
