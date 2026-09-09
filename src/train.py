import torch


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
