def find_misclassified(paths, y_true, y_pred, y_prob):
    misclassified = [
        {"path": path, "true_label": t, "pred_label": p, "prob": prob}
        for path, t, p, prob in zip(paths, y_true, y_pred, y_prob)
        if t != p
    ]
    misclassified.sort(key=lambda r: abs(r["prob"] - 0.5), reverse=True)
    return misclassified
