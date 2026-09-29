"""Trajectory-aggregated metrics; no overlapping window is treated as an independent replicate."""

from __future__ import annotations

import numpy as np


def prediction_error(prediction, target, normalizer):
    """Squared error on original units, with shortest angular difference."""
    scale = normalizer.scale
    delta = (prediction - target) * scale
    delta[:, 4:8] = np.arctan2(np.sin(delta[:, 4:8]), np.cos(delta[:, 4:8]))
    return (delta / normalizer.metric_scale) ** 2


def trajectory_metrics(squared_errors):
    error = np.sqrt(np.mean(np.stack(squared_errors), axis=0))
    return {"nrmse_by_horizon": error.mean(axis=-1).tolist(),
            "mean_nrmse": float(error.mean())}


def balanced_accuracy(true, predicted):
    recalls = []
    for label in (0, 1):
        selected = np.asarray(true) == label
        if selected.any():
            recalls.append(float((np.asarray(predicted)[selected] == label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")
