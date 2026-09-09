import itertools
import random

SEARCH_SPACE = {
    "batch_size": [16, 32, 64],
    "lr": [1e-2, 1e-3, 1e-4],
    "dropout": [0.2, 0.5],
    "patience": [3, 5],
    "l2": [0.0, 1e-4, 1e-3],
    "norm_scheme": ["imagenet", "dataset_stats"],
    "augmentation": ["on", "off"],
}


def sample_configs(search_space, n_trials, seed):
    keys = list(search_space.keys())
    full_grid = list(itertools.product(*(search_space[k] for k in keys)))
    rng = random.Random(seed)
    sampled = rng.sample(full_grid, min(n_trials, len(full_grid)))
    return [dict(zip(keys, combo)) for combo in sampled]
