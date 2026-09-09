import pytest

from src.evaluate import compute_metrics


def test_compute_metrics_matches_hand_computed_confusion_case():
    # 4 NORMAL (0), 4 PNEUMONIA (1); 1 false negative, 1 false positive
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]

    metrics = compute_metrics(y_true, y_pred)

    # class 0: TP=3, FP=1, FN=1 -> precision=0.75, recall=0.75, f1=0.75
    # class 1: TP=3, FP=1, FN=1 -> precision=0.75, recall=0.75, f1=0.75
    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["confusion_matrix"] == [[3, 1], [1, 3]]
    assert metrics["per_class"][0]["precision"] == pytest.approx(0.75)
    assert metrics["per_class"][0]["recall"] == pytest.approx(0.75)
    assert metrics["per_class"][0]["f1"] == pytest.approx(0.75)
    assert metrics["macro"]["precision"] == pytest.approx(0.75)
    assert metrics["macro"]["recall"] == pytest.approx(0.75)
    assert metrics["macro"]["f1"] == pytest.approx(0.75)
