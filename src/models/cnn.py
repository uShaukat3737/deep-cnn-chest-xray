import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    # BN stabilizes training on a dataset this small; MaxPool halves spatial dims each block.
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ChestCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        # Global average pool instead of a flatten+large-FC to keep params low and reduce overfitting.
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def layer_table(model, input_shape):
    rows = []
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append((name, tuple(out.shape), param_count))
        return hook

    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_shape))

    for h in hooks:
        h.remove()

    return rows
