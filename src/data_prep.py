import hashlib
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def discover_images(root):
    root = Path(root)
    return [
        (str(path), path.parent.name)
        for path in root.glob("*/*")
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
