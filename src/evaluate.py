import torch

NUM_CLASSES = 2


@torch.no_grad()
def predict(model, loader, device):
    model.to(device)
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())
    return y_true, y_pred, y_prob


def _confusion_matrix(y_true, y_pred):
    matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def _precision_recall_f1(matrix, cls):
    tp = matrix[cls][cls]
    fp = sum(matrix[r][cls] for r in range(NUM_CLASSES)) - tp
    fn = sum(matrix[cls][c] for c in range(NUM_CLASSES)) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_auc_roc(y_true, y_scores):
    # Mann-Whitney U statistic == AUC-ROC; average ranks handle score ties.
    order = sorted(range(len(y_scores)), key=lambda i: y_scores[i])
    ranks = [0.0] * len(y_scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and y_scores[order[j]] == y_scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    n_pos = sum(1 for t in y_true if t == 1)
    n_neg = len(y_true) - n_pos
    rank_sum_pos = sum(ranks[i] for i in range(len(y_true)) if y_true[i] == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_metrics(y_true, y_pred):
    matrix = _confusion_matrix(y_true, y_pred)
    accuracy = sum(matrix[i][i] for i in range(NUM_CLASSES)) / len(y_true)
    per_class = [_precision_recall_f1(matrix, c) for c in range(NUM_CLASSES)]
    macro = {
        key: sum(pc[key] for pc in per_class) / NUM_CLASSES
        for key in ("precision", "recall", "f1")
    }
    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro": macro,
    }
