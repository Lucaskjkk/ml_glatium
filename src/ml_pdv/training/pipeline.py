"""Pipeline de treinamento de demanda — entrypoint único."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ml_pdv.features.pipeline import build_demand_dataset
from ml_pdv.training.dataset import HORIZON_DAYS, list_tenant_schemas
from ml_pdv.training.split import temporal_train_valid_test_split
from ml_pdv.training.train import metrics_table, pick_best_by_mae, train_model_zoo
from ml_pdv.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass
class TrainingRunSummary:
    tenant: str
    horizon_days: int
    n_rows: int
    n_train: int
    n_valid: int
    n_test: int
    cut_train_end: str
    cut_valid_end: str
    best_model: str
    best_valid_mae: float
    best_test_mae: float
    artifacts_dir: str
    metrics: list[dict]


def run_demand_training(
    *,
    tenant: str | None = None,
    all_tenants: bool = False,
    horizon_days: int = HORIZON_DAYS,
    random_state: int = 42,
    artifacts_dir: str | Path = "artifacts/training",
) -> TrainingRunSummary:
    """
    Fluxo completo:

    dados → features → split temporal → treinar zoo → salvar melhor modelo
    """
    configure_logging()
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_tenants:
        schemas = list_tenant_schemas()
        tenant_label = "all_tenants"
        X, y, meta = build_demand_dataset(tenant_schemas=schemas, horizon_days=horizon_days)
    else:
        if not tenant:
            raise ValueError("Pass --tenant SCHEMA or --all-tenants")
        tenant_label = tenant
        X, y, meta = build_demand_dataset(tenant_schema=tenant, horizon_days=horizon_days)

    if X.empty or len(X) < 10:
        raise RuntimeError(
            f"Dataset too small for training (rows={len(X)}). "
            "Check tenant sales history / dump restore."
        )

    split = temporal_train_valid_test_split(X, y, meta)
    results = train_model_zoo(split, random_state=random_state)
    table = metrics_table(results)
    best = pick_best_by_mae(results, on="valid")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / f"{tenant_label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{best.name}.joblib"
    best.model.save(str(model_path))
    # Também grava um ponteiro "latest" para a API/PredictionService
    latest_path = out_dir / "production_candidate.joblib"
    best.model.save(str(latest_path))
    latest_meta = {
        "model_name": best.name,
        "best_model": best.name,
        "version": stamp,
        "tenant": tenant_label,
        "artifacts_dir": str(run_dir),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "framework": type(best.model).__name__,
        "metrics": {
            "valid_mae": float(best.metrics_valid["mae"]),
            "test_mae": float(best.metrics_test["mae"]),
            "valid_rmse": float(best.metrics_valid["rmse"]),
            "test_rmse": float(best.metrics_test["rmse"]),
        },
    }
    (out_dir / "production_candidate.joblib.meta.json").write_text(
        json.dumps(latest_meta, indent=2),
        encoding="utf-8",
    )

    table.to_csv(run_dir / "metrics.csv", index=False)
    meta_path = run_dir / "feature_columns.json"
    meta_path.write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")

    summary = TrainingRunSummary(
        tenant=tenant_label,
        horizon_days=horizon_days,
        n_rows=len(X),
        n_train=len(split.X_train),
        n_valid=len(split.X_valid),
        n_test=len(split.X_test),
        cut_train_end=str(split.cut_train_end.date()),
        cut_valid_end=str(split.cut_valid_end.date()),
        best_model=best.name,
        best_valid_mae=float(best.metrics_valid["mae"]),
        best_test_mae=float(best.metrics_test["mae"]),
        artifacts_dir=str(run_dir),
        metrics=table.to_dict(orient="records"),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )

    logger.info(
        "training_done",
        tenant=tenant_label,
        best_model=best.name,
        valid_mae=best.metrics_valid["mae"],
        test_mae=best.metrics_test["mae"],
        artifacts=str(run_dir),
    )
    print(table.to_string(index=False))
    print(
        f"\nBest (valid MAE): {best.name} "
        f"valid_mae={best.metrics_valid['mae']:.4f} "
        f"test_mae={best.metrics_test['mae']:.4f}"
    )
    print(f"Saved: {model_path}")
    print(f"Candidate pointer: {latest_path}")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train demand forecasting models")
    parser.add_argument("--tenant", type=str, default=None, help="Ex: tenant_mrcoutinho")
    parser.add_argument("--all-tenants", action="store_true")
    parser.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/training")
    args = parser.parse_args()

    run_demand_training(
        tenant=args.tenant,
        all_tenants=args.all_tenants,
        horizon_days=args.horizon_days,
        random_state=args.random_state,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    main()
