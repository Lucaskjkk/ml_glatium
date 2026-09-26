"""Métricas de regressão para previsão de demanda."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """MAE, RMSE, WAPE, R² (complementar)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "r2": float("nan"), "medae": float("nan")}

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum(np.abs(yt)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > 0 else float("nan")
    medae = float(np.median(np.abs(err)))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "wape": wape, "r2": r2, "medae": medae}
