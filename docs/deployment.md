# Deployment

## Local (current)

```bash
cp .env.example .env
uv sync

# ERP source = dump (default)
# place file at data/dumps/pdv_prod.dump
docker compose up -d postgres-erp
uv run restore-erp-dump
uv run introspect-erp

uv run uvicorn ml_pdv.api.app:app --host 0.0.0.0 --port 8000 --reload
```

OpenAPI: http://localhost:8000/docs

### ERP from dump

| Item | Value |
| --- | --- |
| Mode | `ERP_SOURCE_MODE=dump` |
| Dump path | `ERP_DUMP_PATH=data/dumps/pdv_prod.dump` |
| Local DB | `ERP_DATABASE_URL=postgresql://erp:erp@127.0.0.1:5434/pdv_prod` |

Do **not** use the production bridge IP for local development.

## Docker Compose (current minimal)

Services: `postgres-erp` (dump restore target), `postgres-ml`.

Later: `redis`, `mlflow`, `api`, `worker`.

## AWS (later)

| Local | AWS |
| --- | --- |
| API container | ECS + ECR |
| ML Postgres | RDS |
| Artifacts / Parquet | S3 |
| Logs / metrics | CloudWatch |
| Secrets | Secrets Manager / SSM |

Keep all environment-driven so promotion to AWS does not require a rewrite.
