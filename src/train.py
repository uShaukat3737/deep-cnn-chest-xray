import torch


def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    model.to(device)
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
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
