import pytest

from phishing.evaluation import classification_metrics, risk_level_table


def test_metrics_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = classification_metrics(y_true, y_pred, y_proba=[0.1, 0.6, 0.8, 0.9])
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["false_positive_rate"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 0)


def test_risk_level_table_has_all_levels():
    table = risk_level_table([0, 1], [0.0, 1.0])
    assert table.index.tolist() == ["low", "medium", "high"]
    assert table.loc["medium", "n_urls"] == 0
    assert table.loc["high", "phishing_rate"] == 1.0
