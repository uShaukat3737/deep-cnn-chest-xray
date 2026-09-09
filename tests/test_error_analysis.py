from src.error_analysis import find_misclassified


def test_find_misclassified_filters_and_sorts_by_confidence():
    paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 1, 1]  # b and c are misclassified
    y_prob = [0.1, 0.3, 0.9, 0.6]  # prob of class 1

    result = find_misclassified(paths, y_true, y_pred, y_prob)

    assert [r["path"] for r in result] == ["c.jpg", "b.jpg"]  # sorted by confidence in wrong pred, desc
    assert result[0]["true_label"] == 0
    assert result[0]["pred_label"] == 1
    assert result[0]["prob"] == 0.9
