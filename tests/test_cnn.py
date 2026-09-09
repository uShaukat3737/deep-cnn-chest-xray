import torch

from src.models.cnn import ChestCNN, layer_table


def test_chest_cnn_forward_output_shape():
    model = ChestCNN(num_classes=2, dropout=0.5)
    x = torch.randn(4, 3, 224, 224)

    out = model(x)

    assert out.shape == (4, 2)


def test_layer_table_reports_shapes_and_param_counts():
    model = ChestCNN(num_classes=2, dropout=0.5)

    rows = layer_table(model, input_shape=(1, 3, 224, 224))

    assert len(rows) > 0
    for name, output_shape, param_count in rows:
        assert isinstance(name, str)
        assert isinstance(output_shape, tuple)
        assert isinstance(param_count, int)
        assert param_count >= 0
    total_params = sum(p.numel() for p in model.parameters())
    assert sum(r[2] for r in rows) == total_params
