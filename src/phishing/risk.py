"""Risk score (1 to 10) and risk level built from the phishing probability."""

import numpy as np
from numpy.typing import ArrayLike

from phishing import config


def risk_score(p_phishing: ArrayLike) -> np.ndarray:
    """Risk score = 1 + 9 * P(phishing), so 1 is safe and 10 is surely phishing."""
    p = np.asarray(p_phishing, dtype=float)
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError("probabilities must be in [0, 1]")
    return 1.0 + 9.0 * p


def risk_level(score: ArrayLike) -> np.ndarray:
    """Map scores to "low" [1, 4), "medium" [4, 7) and "high" [7, 10]."""
    s = np.asarray(score, dtype=float)
    if np.any((s < 1.0) | (s > 10.0)) or np.any(np.isnan(s)):
        raise ValueError("scores must be in [1, 10]")
    names = np.array([name for name, _ in config.RISK_LEVELS])
    thresholds = np.array([low for _, low in config.RISK_LEVELS])
    return names[np.searchsorted(thresholds, s, side="right") - 1]
