import torch.nn as nn
from torchvision import models


def _build_resnet50(pretrained, num_classes):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenetv2(pretrained, num_classes):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


_HEAD_PREFIX = {"resnet50": "fc.", "mobilenetv2": "classifier.1."}
_BUILDERS = {"resnet50": _build_resnet50, "mobilenetv2": _build_mobilenetv2}


def build_model(name, mode, num_classes=2, pretrained=True):
    if name not in _BUILDERS:
        raise ValueError(f"unknown model: {name}")
    model = _BUILDERS[name](pretrained, num_classes)

    if mode == "frozen":
        head_prefix = _HEAD_PREFIX[name]
        for n, p in model.named_parameters():
            if not n.startswith(head_prefix):
                p.requires_grad = False
    elif mode != "finetune":
        raise ValueError(f"unknown mode: {mode}")

    return model
