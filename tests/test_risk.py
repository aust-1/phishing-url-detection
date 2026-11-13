import numpy as np
import pytest

from phishing.risk import risk_level, risk_score


def test_score_bounds():
    np.testing.assert_allclose(risk_score([0.0, 0.5, 1.0]), [1.0, 5.5, 10.0])


@pytest.mark.parametrize("p", [-0.1, 1.1, np.nan])
def test_score_rejects_bad_probabilities(p):
    with pytest.raises(ValueError):
        risk_score([p])


def test_levels_edges():
    levels = risk_level([1.0, 3.99, 4.0, 6.99, 7.0, 10.0])
    assert levels.tolist() == ["low", "low", "medium", "medium", "high", "high"]


def test_levels_reject_out_of_range():
    with pytest.raises(ValueError):
        risk_level([0.5])
