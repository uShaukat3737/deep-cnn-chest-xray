from PIL import Image

from src.data_prep import dedupe_by_hash, filter_corrupt, stratified_split


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
