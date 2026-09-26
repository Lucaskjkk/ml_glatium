# ML PDV Platform

Plataforma de Machine Learning (monólito modular) para **previsão de demanda/vendas** a partir do PostgreSQL do ERP — desacoplada, multi-tenant e preparada para outros modelos (churn, recomendação, estoque, etc.).

## Objetivo

- Consumir o ERP em **READ ONLY**
- Copiar dados para um **banco analítico ML**
- Features → treino → MLflow → registry → **API de predição**
- Falhas no ML **não** afetam o ERP

## Arquitetura (resumo)

```text
ERP dump (.dump/.sql)
        ↓ restore-erp-dump
Postgres local (postgres-erp :5434)  →  Ingestão incremental  →  ML Postgres
                                                                      ↓
                                                            Treino + MLflow
                                                                      ↓
                                                            FastAPI → ERP/Dashboard
```

Detalhes: [docs/architecture.md](docs/architecture.md)

## Stack

Python 3.12+, FastAPI, Pydantic Settings, SQLAlchemy, Polars/Pandas, scikit-learn, XGBoost, MLflow (próximas fases), Celery/Redis (próximas), Docker.

## Setup rápido

```bash
cp .env.example .env
uv sync
```

### Fonte de dados do ERP = dump (padrão)

O projeto **não** depende mais da conexão live `172.x`. Fluxo:

```bash
# 1) Gere o dump (em máquina com acesso ao ERP)
pg_dump -Fc -h ERP_HOST -U USER -d pdv_prod -f data/dumps/pdv_prod.dump

# 2) Suba o Postgres local e restaure
docker compose up -d postgres-erp
uv run restore-erp-dump

# 3) Mapeie o schema
uv run introspect-erp

# 4) Treine demanda (baseline + RF + XGBoost)
uv run train-demand --tenant tenant_mrcoutinho

# 5) API
uv run uvicorn ml_pdv.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Walkthrough do treino: [docs/training-walkthrough.md](docs/training-walkthrough.md)

- Health: http://localhost:8000/health  
- OpenAPI: http://localhost:8000/docs  

## Variáveis de ambiente

Veja [.env.example](.env.example). Principais:

| Variável | Uso |
| --- | --- |
| `ERP_SOURCE_MODE` | `dump` (padrão) ou `live` |
| `ERP_DUMP_PATH` | Caminho do `.dump` / `.sql` |
| `ERP_DATABASE_URL` | Postgres fonte (local restaurado ou live) |
| `ML_DATABASE_URL` | Postgres analítico ML |
| `API_KEY` | Header `X-API-Key` |
| `MODEL_NAME` | Nome no registry (`demand_forecasting`) |
| `PREDICTION_HORIZON_DAYS` | Horizonte padrão (7) |

**Nunca** commite `.env` nem dumps em `data/dumps/`.

## API (v1)

| Método | Path | Auth |
| --- | --- | --- |
| GET | `/health` | não |
| GET | `/ready` | não |
| GET | `/api/v1/models` | `X-API-Key` |
| GET | `/api/v1/models/production` | `X-API-Key` |
| POST | `/api/v1/predictions/demand` | `X-API-Key` |
| GET | `/api/v1/predictions/products/{product_id}?tenant_id=` | `X-API-Key` |

Exemplo de predição (enquanto não houver modelo em produção, responde **503** com `model_status=not_ready` — contratos já estáveis para integrar no ERP):

```bash
curl -s -X POST http://localhost:8000/api/v1/predictions/demand \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change-me-in-production' \
  -d '{"tenant_id":"empresa-1","product_ids":["SKU-1","SKU-2"]}'
```

## Banco ML

Schemas lógicos: `raw`, `clean`, `features`, `predictions`, `metadata` (sync incremental).  
Ver [docs/data-pipeline.md](docs/data-pipeline.md).

## Documentação

| Doc | Conteúdo |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Arquitetura |
| [docs/erp-schema-map.md](docs/erp-schema-map.md) | Mapa real do ERP |
| [docs/data-pipeline.md](docs/data-pipeline.md) | Ingestão + ML DB |
| [docs/model-lifecycle.md](docs/model-lifecycle.md) | Treino → registry → serve |
| [docs/improving-predictions.md](docs/improving-predictions.md) | **O que mudar depois** para melhorar / expandir |
| [docs/deployment.md](docs/deployment.md) | Local → AWS |

## Segurança

- Credenciais só em `.env`
- Preferir user READ ONLY no ERP
- Se uma senha foi compartilhada em chat/canal, **rotacione**
- Minimizar PII nas features

## Status da Fase 1

- [x] Configuração + settings + `.env`
- [x] Scaffold modular + FastAPI
- [x] Script de introspecção
- [x] Docs de arquitetura / pipeline / evolução
- [ ] Schema ERP preenchido (aguardando conectividade de rede ao Postgres do ERP)
- [ ] Ingestão / treino / MLflow / Celery / Compose (próximas fases)

## Desenvolvimento

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```
