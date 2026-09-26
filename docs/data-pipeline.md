# Data Pipeline & ML Database

## Principles

1. ERP PostgreSQL is **read-only** for ML.
2. All analytical copies live in a **separate** ML PostgreSQL.
3. Sync is **incremental** and **idempotent**, keyed by `(tenant_id, source_pk)`.
4. Multi-tenant: every business row carries `tenant_id` (mapped from the real ERP column after discovery).

## Incremental ingestion strategy

### Control table: `metadata.sync_state`

| Column | Purpose |
| --- | --- |
| `tenant_id` | Tenant scope |
| `source_table` | ERP table logical name |
| `watermark_column` | Column used for incremental filter |
| `last_sync_timestamp` | Last successful watermark |
| `last_processed_id` | Tie-breaker PK / cursor |
| `status` | `idle` / `running` / `failed` |
| `error_message` | Last error |
| `updated_at` | Checkpoint update time |

Unique: `(tenant_id, source_table)`.

### Sync algorithm (per tenant × table)

```text
1. READ sync_state for (tenant_id, source_table)
2. SELECT batch FROM erp.source
     WHERE watermark > :last_sync_timestamp
        OR (watermark = :last_sync_timestamp AND pk > :last_processed_id)
     ORDER BY watermark, pk
     LIMIT :batch_size
   [AND tenant_col = :tenant_id]
3. UPSERT into raw.* ON CONFLICT (tenant_id, source_pk) DO UPDATE
4. COMMIT batch
5. UPDATE sync_state watermarks  -- only after successful commit
6. Repeat until empty
```

### Watermark preference

1. `updated_at` (or ERP equivalent) + PK — handles inserts and updates.
2. Only `created_at` — append-only; schedule periodic reconciliation windows.
3. No timestamp — PK ranges + periodic full reconcile (document cost).

### Idempotency & failures

- At-least-once delivery + UPSERT = no duplicate raw rows.
- Job retry safe; do not advance checkpoint before commit.
- Soft deletes: propagate `deleted_at` / `ativo` when present in ERP.

### Scheduling (later with Celery)

| Job | Cadence |
| --- | --- |
| Ingestion | Every few minutes |
| Feature generation | After ingestion / periodic |
| Training | Weekly (not per sale) |
| Prediction | On-demand API + optional batch |

## ML database schemas

| Schema | Role |
| --- | --- |
| `raw` | Near-mirror of selected ERP tables + `tenant_id` + ingest metadata |
| `clean` | Typed, validated, business-rule filtered tables |
| `features` | ML-ready feature tables (versionable; later also Parquet/DVC) |
| `predictions` | Stored inference outputs for audit / ERP fetch |
| `metadata` | `sync_state`, `pipeline_runs`, quality metrics, dataset versions |

### Example physical tables (names finalized after ERP map)

```text
raw.sales
raw.sale_items
raw.products
raw.customers
raw.stock

clean.sales
clean.sale_items
clean.products

features.demand_daily          -- grain: (tenant_id, product_id, feature_as_of)
predictions.demand             -- grain: (tenant_id, product_id, as_of, horizon, model_version)
metadata.sync_state
metadata.pipeline_runs
metadata.data_quality_metrics
metadata.dataset_versions
```

### Suggested indexes

- `(tenant_id, <business_date>)` on sales/clean sales
- `(tenant_id, product_id, feature_as_of)` on `features.demand_daily`
- `(tenant_id, product_id, created_at DESC)` on `predictions.demand`
- Unique `(tenant_id, source_pk)` on every `raw.*` table

## Demand forecast contract (7 days)

| Concept | Definition |
| --- | --- |
| `feature_as_of` (D) | Cutoff date — features use only data with event_time ≤ D |
| Prediction window | Sales quantity from D+1 through D+7 inclusive |
| Grain | `(tenant_id, product_id, feature_as_of)` |
| Target | Sum of sold quantity in the prediction window |

**Anti-leakage:** no feature may use information available only after D.

Example:

```text
features up to 2026-09-16
→ predict quantity sold 2026-09-17 … 2026-09-23
```

## Temporal split (no random shuffle)

Prefer calendar or size-based temporal holdout, e.g.:

- train: oldest 70% of time range
- validation: next 15%
- test: latest 15%

Also support walk-forward / `TimeSeriesSplit` for model selection.

## Data quality gates

Before features/training:

- null rates, type checks, duplicates
- invalid dates, negative quantities/prices when forbidden
- orphan product/customer FKs
- sales without items

Configurable fail threshold via settings (e.g. `DATA_QUALITY_MIN_SCORE`).

## Future: Parquet / S3 / DVC

Keep processing APIs file-oriented (Polars DataFrames / Parquet paths) so later:

```text
PostgreSQL → S3 Parquet → feature pipelines → MLflow/DVC
```

without rewriting business logic.
