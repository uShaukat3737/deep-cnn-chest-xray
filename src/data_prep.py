import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def discover_images(root):
    root = Path(root)
    return [
        (str(path), path.parent.name)
        for path in root.rglob("*")
        if path.is_file()
    ]


def dedupe_by_hash(items):
    seen = set()
    result = []
    for path, label in items:
        with Image.open(path) as img:
            digest = hashlib.md5(img.convert("RGB").tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        result.append((path, label))
    return result


def filter_corrupt(items):
    result = []
    for path, label in items:
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            continue
        result.append((path, label))
    return result


def stratified_split(items, seed, ratios=(0.8, 0.1, 0.1)):
    by_label = {}
    for path, label in items:
        by_label.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label_items in by_label.values():
        shuffled = label_items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    return train, val, test


def subsample_by_fraction(items, fraction, seed):
    by_label = {}
    for path, label in items:
        by_label.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    result = []
    for label_items in by_label.values():
        shuffled = label_items[:]
        rng.shuffle(shuffled)
        n = round(len(shuffled) * fraction)
        result.extend(shuffled[:n])
    return result


def _write_manifest(path, items):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(items)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    items = discover_images(args.data_dir)
    items = filter_corrupt(items)
    items = dedupe_by_hash(items)
    train, val, test = stratified_split(items, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_items in (("train", train), ("val", val), ("test", test)):
        _write_manifest(out_dir / f"{split_name}.csv", split_items)
        print(f"{split_name}: {len(split_items)} ({dict(Counter(l for _, l in split_items))})")


if __name__ == "__main__":
    main()
