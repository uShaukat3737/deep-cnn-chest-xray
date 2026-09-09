from src.data_prep import stratified_split


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
