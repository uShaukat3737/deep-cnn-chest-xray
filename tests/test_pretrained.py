import torch

from src.models.pretrained import build_model


def test_frozen_resnet50_freezes_backbone_but_not_head():
    model = build_model("resnet50", mode="frozen", pretrained=False, num_classes=2)

    head_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]

    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in head_params)

    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)
