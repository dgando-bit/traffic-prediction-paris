from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shared.logging import get_logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

logger = get_logger(__name__)


@dataclass
class RegressionMetrics:
    """Regression evaluation metrics."""

    mae: float
    rmse: float
    n_samples: int


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """
    Evaluate regression predictions using MAE and RMSE.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate empty predictions."
        )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    metrics = RegressionMetrics(
        mae=float(mae),
        rmse=float(rmse),
        n_samples=len(y_true),
    )

    logger.info(
        "Regression evaluation - MAE: %.4f | RMSE: %.4f | n=%s",
        metrics.mae,
        metrics.rmse,
        metrics.n_samples,
    )

    return metrics