"""Split temporal — nunca use random shuffle em séries de demanda."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    meta_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_valid: pd.Series
    meta_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_test: pd.DataFrame
    cut_train_end: pd.Timestamp
    cut_valid_end: pd.Timestamp


def temporal_train_valid_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    date_col: str = "feature_as_of",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> TemporalSplit:
    """
    Divide por tempo global (todas as linhas ordenadas por feature_as_of).

    Ex.: 70% treino / 15% validação / 15% teste no eixo temporal.
    """
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1 and positive")

    order = meta[date_col].argsort(kind="mergesort")
    X_o = X.iloc[order].reset_index(drop=True)
    y_o = y.iloc[order].reset_index(drop=True)
    m_o = meta.iloc[order].reset_index(drop=True)

    n = len(m_o)
    if n < 10:
        raise ValueError(f"Not enough rows for temporal split: {n}")

    i_train = int(n * train_ratio)
    i_valid = int(n * (train_ratio + valid_ratio))
    i_train = max(i_train, 1)
    i_valid = max(i_valid, i_train + 1)
    i_valid = min(i_valid, n - 1)

    cut_train_end = pd.to_datetime(m_o.iloc[i_train - 1][date_col])
    cut_valid_end = pd.to_datetime(m_o.iloc[i_valid - 1][date_col])

    return TemporalSplit(
        X_train=X_o.iloc[:i_train],
        y_train=y_o.iloc[:i_train],
        meta_train=m_o.iloc[:i_train],
        X_valid=X_o.iloc[i_train:i_valid],
        y_valid=y_o.iloc[i_train:i_valid],
        meta_valid=m_o.iloc[i_train:i_valid],
        X_test=X_o.iloc[i_valid:],
        y_test=y_o.iloc[i_valid:],
        meta_test=m_o.iloc[i_valid:],
        cut_train_end=cut_train_end,
        cut_valid_end=cut_valid_end,
    )
