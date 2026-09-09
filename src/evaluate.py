NUM_CLASSES = 2


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
