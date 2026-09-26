# Model Lifecycle

## Stages

```text
Train → Evaluate → Compare vs production → Approve/Reject → Registry → Serve
```

Promotion is **not** automatic just because a new run finished.

## Tracking (MLflow)

Each training run should log:

- parameters / hyperparameters
- feature list + feature version
- dataset version / commit hash
- metrics (MAE, RMSE, WAPE, MAPE when valid)
- model artifact
- random seed
- pipeline version / code revision

Experiment name example: `demand_forecasting`.

## Registry

Registered model name: `demand_forecasting` (configurable via `MODEL_NAME`).

Stages: `candidate` → `staging` → `production` (or reject).

## Promotion policy (configurable)

Example rule:

```text
IF candidate.MAE <= production.MAE * (1 - improvement_threshold)
   AND candidate passes validation gates
THEN promote to production
ELSE mark rejected
```

Defaults live in settings / configs — never hardcode magic numbers in route handlers.

## Serving

`PredictionService`:

- loads production model **once** (process lifetime)
- validates inputs / outputs
- returns `model_version` on every response
- returns `503 model_status=not_ready` when no production model exists

Training failure must leave the previous production model untouched.

## Dataset lineage

Store in `metadata.dataset_versions` (and later DVC):

```text
model v3 ← dataset 2026-09-16 ← features 2.1 ← git abc123
```

## Baselines first

Always compare ML models against:

- naive (last value)
- moving average
- historical mean

A model that does not beat baseline on holdout is not production-ready.
