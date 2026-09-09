import torch.nn as nn
from torchvision import models


def _build_resnet50(pretrained, num_classes):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name, mode, num_classes=2, pretrained=True):
    if name == "resnet50":
        model = _build_resnet50(pretrained, num_classes)
    else:
        raise ValueError(f"unknown model: {name}")

    if mode == "frozen":
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    elif mode != "finetune":
        raise ValueError(f"unknown mode: {mode}")

    return model
