# Architecture

## Goal

Modular ML platform for demand/sales forecasting (and later churn, recommendations, etc.) on real ERP sales data, without coupling to or mutating the ERP transactional database.

## High-level flow

```text
ERP dump (.dump / .sql)
        │  uv run restore-erp-dump
        ▼
 Local Postgres (postgres-erp) ──READ──► Ingestion ──► ML PostgreSQL
                                                              │
                                    Training / MLflow ◄───────┤
                                                              │
                                    PredictionService / FastAPI
```

Local default: `ERP_SOURCE_MODE=dump` (no live production connection).
Optional later: `ERP_SOURCE_MODE=live` with a READ ONLY URL.

## Modular monolith

Single Python package `ml_pdv` with clear boundaries:

| Module | Responsibility |
| --- | --- |
| `api/` | HTTP adapters only (FastAPI routes) |
| `inference/` | Prediction orchestration |
| `models/` | Model interfaces + registry facade |
| `ingestion/` | ERP → ML sync |
| `data/` | Validation / cleaning |
| `features/` | Feature engineering |
| `training/` | Train / evaluate / promote |
| `monitoring/` | Drift, quality, performance |
| `jobs/` | Celery tasks (later) |
| `database/` | Engines, ORM metadata |
| `config/` | Pydantic Settings |

Future split into microservices is possible **without** rewriting domain logic if these boundaries stay clean.

## Isolation rules

- Failures in ML (API down, training crash, MLflow down) **must not** affect ERP transactions.
- Training never blocks inference; API keeps serving the last production model.
- ERP credentials: least privilege, READ ONLY.

## Multi-tenant

- Product already has multiple clients/empresas.
- `tenant_id` is mandatory on ML analytical tables and on API requests.
- Sync and features always filtered by tenant.

## API surface (v1)

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness / degraded if model missing |
| GET | `/api/v1/models` | Registered models |
| GET | `/api/v1/models/production` | Current production model |
| POST | `/api/v1/predictions/demand` | Demand forecast (503 until model ready) |
| GET | `/api/v1/predictions/products/{id}` | Per-product prediction view |

Auth: `X-API-Key` header (env `API_KEY`). Swap for JWT later without changing route paths.

## Cloud readiness (not implemented yet)

Local Compose → later AWS: ECS/ECR, RDS (ML DB), S3 (Parquet), CloudWatch. Keep config via environment variables only.

## See also

- [erp-schema-map.md](erp-schema-map.md)
- [data-pipeline.md](data-pipeline.md)
- [model-lifecycle.md](model-lifecycle.md)
- [improving-predictions.md](improving-predictions.md)
- [deployment.md](deployment.md)
