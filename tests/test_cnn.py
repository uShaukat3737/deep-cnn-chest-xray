import torch

from src.models.cnn import ChestCNN


def test_chest_cnn_forward_output_shape():
    model = ChestCNN(num_classes=2, dropout=0.5)
    x = torch.randn(4, 3, 224, 224)

    out = model(x)

    assert out.shape == (4, 2)
