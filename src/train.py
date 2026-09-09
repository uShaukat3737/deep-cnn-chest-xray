import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import ChestXrayDataset, build_sample_weights, build_transforms
from src.models.cnn import ChestCNN
from src.models.pretrained import build_model


def l1_penalty(model, lam):
    return lam * sum(p.abs().sum() for p in model.parameters())


def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0, l1_lambda=0.0):
    model.to(device)
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y) + l1_penalty(model, l1_lambda)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


class Checkpointer:
    def __init__(self, path):
        self.path = path
        self.best = float("inf")

    def step(self, model, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            torch.save(model.state_dict(), self.path)
            return True
        return False


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best = float("inf")
        self.num_bad_epochs = 0

    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def _build_model(name, mode, dropout):
    if name == "cnn":
        return ChestCNN(num_classes=2, dropout=dropout)
    return build_model(name, mode=mode, num_classes=2)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "resnet50", "mobilenetv2"])
    parser.add_argument("--mode", required=True, choices=["scratch", "frozen", "finetune"])
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tf, val_tf = build_transforms()
    train_ds = ChestXrayDataset(args.train_manifest, transform=train_tf)
    val_ds = ChestXrayDataset(args.val_manifest, transform=val_tf)

    labels = [label for _, label in train_ds.items]
    weights = build_sample_weights(labels)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = _build_model(args.model, args.mode, args.dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, args.patience // 2))
    early_stopper = EarlyStopping(patience=args.patience)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.model}_{args.mode}_best.pth"
    checkpointer = Checkpointer(checkpoint_path)

    with open(args.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            checkpointer.step(model, val_loss)
            writer.writerow([epoch, train_loss, val_loss, val_acc])
            f.flush()

            if early_stopper.step(val_loss):
                break


if __name__ == "__main__":
    main()
