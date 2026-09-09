from PIL import Image

import csv

from src.data_prep import (
    dedupe_by_hash,
    discover_images,
    filter_corrupt,
    main,
    stratified_split,
    subsample_by_fraction,
)


def test_stratified_split_ratios_and_no_overlap():
    items = [(f"normal_{i}.jpg", "NORMAL") for i in range(100)] + [
        (f"pneu_{i}.jpg", "PNEUMONIA") for i in range(300)
    ]

    train, val, test = stratified_split(items, seed=42, ratios=(0.8, 0.1, 0.1))

    train_paths = {p for p, _ in train}
    val_paths = {p for p, _ in val}
    test_paths = {p for p, _ in test}
    assert not (train_paths & val_paths)
    assert not (train_paths & test_paths)
    assert not (val_paths & test_paths)
    assert len(train) + len(val) + len(test) == len(items)

    for label in ("NORMAL", "PNEUMONIA"):
        total = sum(1 for _, l in items if l == label)
        train_count = sum(1 for _, l in train if l == label)
        assert train_count == round(total * 0.8)


def test_filter_corrupt_excludes_unreadable_files(tmp_path):
    good = tmp_path / "good.jpg"
    Image.new("RGB", (10, 10), "white").save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")

    items = [(str(good), "NORMAL"), (str(bad), "NORMAL")]
    result = filter_corrupt(items)

    assert result == [(str(good), "NORMAL")]


def test_dedupe_by_hash_drops_identical_pixel_content(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.jpg"
    Image.new("RGB", (10, 10), "white").save(a)
    Image.new("RGB", (10, 10), "white").save(b)
    Image.new("RGB", (10, 10), "black").save(c)

    items = [(str(a), "NORMAL"), (str(b), "NORMAL"), (str(c), "NORMAL")]
    result = dedupe_by_hash(items)

    assert len(result) == 2
    assert (str(a), "NORMAL") in result
    assert (str(c), "NORMAL") in result


def test_discover_images_labels_by_parent_dir(tmp_path):
    normal_dir = tmp_path / "NORMAL"
    pneu_dir = tmp_path / "PNEUMONIA"
    normal_dir.mkdir()
    pneu_dir.mkdir()
    (normal_dir / "n1.jpg").touch()
    (pneu_dir / "p1.jpg").touch()
    (pneu_dir / "p2.jpg").touch()

    result = discover_images(tmp_path)

    assert sorted(result) == sorted([
        (str(normal_dir / "n1.jpg"), "NORMAL"),
        (str(pneu_dir / "p1.jpg"), "PNEUMONIA"),
        (str(pneu_dir / "p2.jpg"), "PNEUMONIA"),
    ])


def test_main_writes_split_manifests(tmp_path):
    normal_dir = tmp_path / "NORMAL"
    pneu_dir = tmp_path / "PNEUMONIA"
    normal_dir.mkdir()
    pneu_dir.mkdir()
    for i in range(10):
        Image.new("RGB", (5, 5), "white").save(normal_dir / f"n{i}.jpg")
    for i in range(10):
        Image.new("RGB", (5, 5), (i, 0, 0)).save(pneu_dir / f"p{i}.jpg")

    out_dir = tmp_path / "manifests"
    main(["--data-dir", str(tmp_path), "--out-dir", str(out_dir), "--seed", "42"])

    for split in ("train", "val", "test"):
        with open(out_dir / f"{split}.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["path", "label"]

    with open(out_dir / "train.csv") as f:
        train_rows = list(csv.reader(f))[1:]
    with open(out_dir / "test.csv") as f:
        test_rows = list(csv.reader(f))[1:]
    assert not ({r[0] for r in train_rows} & {r[0] for r in test_rows})


def test_subsample_by_fraction_preserves_class_ratio_and_is_deterministic():
    items = [(f"n{i}.jpg", "NORMAL") for i in range(40)] + [
        (f"p{i}.jpg", "PNEUMONIA") for i in range(80)
    ]

    result_a = subsample_by_fraction(items, fraction=0.5, seed=42)
    result_b = subsample_by_fraction(items, fraction=0.5, seed=42)

    assert result_a == result_b
    assert sum(1 for _, l in result_a if l == "NORMAL") == 20
    assert sum(1 for _, l in result_a if l == "PNEUMONIA") == 40
