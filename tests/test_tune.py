from src.tune import sample_configs


def test_sample_configs_returns_unique_deterministic_configs():
    search_space = {
        "batch_size": [16, 32, 64],
        "lr": [1e-2, 1e-3, 1e-4],
        "dropout": [0.2, 0.5],
    }

    configs_a = sample_configs(search_space, n_trials=5, seed=42)
    configs_b = sample_configs(search_space, n_trials=5, seed=42)

    assert configs_a == configs_b
    assert len(configs_a) == 5

    seen = {tuple(sorted(c.items())) for c in configs_a}
    assert len(seen) == 5

    for config in configs_a:
        assert set(config.keys()) == {"batch_size", "lr", "dropout"}
        assert config["batch_size"] in search_space["batch_size"]
