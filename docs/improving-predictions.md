# Improving Predictions & Extending the Platform

This guide explains **what you can change later** to improve demand forecasts and add new ML capabilities — without rewriting the whole system.

---

## 1. Data & connectivity (highest leverage first)

| Change | Why it helps |
| --- | --- |
| Publish ERP Postgres to a host-reachable address / same Docker network | Unlocks real schema map + incremental sync |
| Create a **READ ONLY** DB user for ML | Safer than `root`; least privilege |
| Re-run `uv run introspect-erp` after schema changes | Keeps `docs/erp-schema-map.md` truthful |
| Fix soft-deletes / cancelled sales filters in `clean` | Removes noise from training targets |
| Ensure reliable `updated_at` (or equivalent) on sales tables | Makes incremental sync correct |

Bad labels beat fancy models. Prefer cleaning sales status and item quantities before tuning XGBoost.

---

## 2. Features you can add for demand forecasting

Implement in `ml_pdv/features/` (e.g. `sales.py`, `products.py`, `pipeline.py`). Keep grain `(tenant_id, product_id, feature_as_of)`.

Useful additions:

- Rolling windows: 7 / 14 / 30 / 60 / 90 day sales sum, mean, std
- Price and discount features (as-of D only)
- Stock on hand / days of cover (if ERP has stock)
- Calendar: day of week, week of year, month, holidays, payday effects
- Product attributes: category, brand, ABC class
- Promo flags / campaign calendars
- Store / filial effects (still within tenant)

**Rule:** never use data after `feature_as_of` (no leakage).

To change horizon (14d, 30d): update `PREDICTION_HORIZON_DAYS` and rebuild targets in the training dataset builder — keep API field `horizon_days` so clients can request supported horizons.

---

## 3. Models & baselines

| What to change | Where |
| --- | --- |
| Add a new algorithm | Implement `BaseForecastModel` in `ml_pdv/models/` |
| Compare fair | Always log baseline metrics in MLflow |
| Hyperparameters | Config YAML under `configs/` + MLflow params |
| Ensemble | New model class that wraps others — keep registry name clear |

Suggested order: baselines → RandomForest → XGBoost → LightGBM → only then deeper models if needed.

---

## 4. Training, validation, promotion

| Lever | Action |
| --- | --- |
| Temporal split | Adjust ranges in `training/split.py` (never random shuffle for time series) |
| Walk-forward | Enable TimeSeriesSplit for model selection |
| Metrics | MAE / RMSE / WAPE primary; MAPE only when zeros are rare |
| Promotion gate | Configure improvement threshold vs production MAE |
| Retrain cadence | Weekly Celery beat — **not** on every sale |

Wire MLflow Model Registry so `PredictionService` loads the production stage once per process.

---

## 5. Serving & API integration (ERP side)

Stable contracts under `/api/v1`:

- `POST /api/v1/predictions/demand` with `tenant_id` + `product_ids`
- Header `X-API-Key`
- Response includes `model_version` and `model_status`

When a model is promoted, ERP keeps the same endpoints — only numbers and `model_version` change.

Later improvements:

- Persist predictions to `predictions.demand` for offline dashboards
- Batch prediction job for all SKUs per tenant overnight
- JWT / per-tenant auth instead of shared API key
- Rate limiting and idempotency keys

---

## 6. Adding entirely new ML products

Use the same modular layout; do **not** fork the repo.

| Product | New pieces |
| --- | --- |
| Stock forecast | Features from stock movements + demand forecast as input |
| Customer churn | `features/customers.py`, binary classification model, new route `/predictions/churn` |
| Repurchase probability | Similar to churn; different target window |
| Product recommendation | Ranking model; endpoint under `/api/v1/recommendations` |
| Segmentation | Clustering job; write segments to `features.customer_segments` |
| Anomaly detection | Streaming/batch scores into `predictions.anomalies` |

Pattern:

1. New feature module + dataset builder
2. New `BaseForecastModel` (or classifier) implementation
3. New MLflow experiment name
4. New FastAPI router calling a dedicated service (no logic in routes)
5. Document target definition and leakage rules

Keep `tenant_id` everywhere.

---

## 7. Data versioning & reproducibility (DVC / Parquet / S3)

When datasets grow:

1. Export feature tables to Parquet (Polars)
2. Track with **DVC** (dataset versions) while **MLflow** tracks experiments/models
3. Later move Parquet to S3; pipelines read paths, not “whatever is in memory”

Always log: seed, feature version, dataset version, git commit.

---

## 8. Quality, monitoring, retraining triggers

Add over time in `ml_pdv/monitoring/`:

- Data quality score gates (fail training if below threshold)
- Feature / prediction drift
- Online vs offline metric gaps
- Pipeline failure alerts

Possible retrain triggers: scheduled weekly, or drift above threshold — still **compare to production** before promote.

---

## 9. Ops & scale (when you outgrow the monolith)

Keep the modular monolith until a real bottleneck appears. Then extract:

- `data-ingestion-service`
- `feature-service`
- `training-service`
- `prediction-service`

Infra path: Docker Compose → ECS/ECR + RDS + S3 + CloudWatch.

---

## 10. What you should *not* change casually

- Do **not** add ML tables or triggers inside ERP production DB
- Do **not** train on live transactional tables directly
- Do **not** put secrets in code or commit `.env`
- Do **not** auto-replace production models without a gate
- Do **not** mix training code into FastAPI route handlers

---

## Practical “next 5” checklist

1. Make ERP reachable → `uv run introspect-erp`
2. Stand up ML Postgres + Alembic for `raw` / `metadata`
3. Implement incremental sync for sales + items + products
4. Build demand features + baseline + first XGBoost run in MLflow
5. Promote model → `POST /predictions/demand` returns 200 with real numbers
