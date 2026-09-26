# Walkthrough do treinamento de demanda

Este doc explica **o que cada arquivo faz**, **por que existe** e **o que se repete** se você criar outro modelo (churn, estoque, etc.).

## Como rodar

```bash
# um tenant
uv run train-demand --tenant tenant_mrcoutinho

# todos os tenants do dump
uv run train-demand --all-tenants
```

Artefatos em `artifacts/training/`:
- `metrics.csv` — comparação baseline vs ML
- `<best>.joblib` — modelo vencedor (menor MAE na validação)
- `production_candidate.joblib` — ponteiro rápido para o melhor da última run
- `summary.json` — metadados do experimento

## Fluxo mental

```text
ERP dump (local)
  → dataset.load_daily_sales   (fato bruto: produto × dia × qty)
  → densify                     (preenche dias sem venda com 0)
  → add_target                  (y = soma qty D+1..D+7)
  → features                    (X sem usar futuro)
  → split temporal              (treino → valid → teste no tempo)
  → train zoo                   (baseline + RF + XGBoost)
  → evaluate                    (MAE / RMSE / WAPE)
  → salva melhor modelo
```

## Arquivo por arquivo

### 1. `src/ml_pdv/training/dataset.py`

| Função | Por quê |
| --- | --- |
| `HORIZON_DAYS` / `TARGET_COL` | Contrato único do problema (API e treino alinhados) |
| `_validate_tenant_schema` | Schema entra no SQL como identificador → bloqueia injection |
| `load_daily_sales` | Lê fato de negócio; filtra `status='finalizada'` |
| `densify_daily_sales` | Sem isso, `shift(-1)` pula para a *próxima venda*, não o *próximo dia* |
| `add_target` | Soma qty futura; remove linhas sem horizonte completo |
| `build_supervised_frame` | Junta densify + target + preço/categoria |

**Repete em outros modelos?** Sim a estrutura (load → limpa → target). Muda grain e fórmula do y.

### 2. `src/ml_pdv/features/sales.py`

| Função | Por quê |
| --- | --- |
| `add_lag_rolling_features` | Memória de curto/médio prazo (7/14/30d) **só com passado ≤ D** |
| `add_calendar_features` | Sazonalidade semanal/mensal |
| `build_feature_matrix` | Separa X / y / meta (ids não vão para o modelo) |

**Anti-leakage:** target usa D+1..D+7; features usam qty até D inclusive.

### 3. `src/ml_pdv/features/pipeline.py`

`build_demand_dataset` — cola dataset + features. Um ponto de entrada para o treino.

### 4. `src/ml_pdv/training/split.py`

Split **temporal** (70/15/15).  
**Nunca** `train_test_split(shuffle=True)` em demanda — vaza futuro no treino.

### 5. `src/ml_pdv/training/evaluate.py`

MAE (principal), RMSE, WAPE, R² complementar.  
WAPE é útil quando há muitos zeros (comum em SKU × dia).

### 6. `src/ml_pdv/models/demand_forecasting.py`

| Modelo | Papel |
| --- | --- |
| `naive_last_value` | Baseline fraco (sanity) |
| `moving_average_7` | Baseline forte (média × horizonte) |
| `random_forest` | ML clássico |
| `xgboost` | ML boosting |

Todo modelo implementa `BaseForecastModel` (`fit/predict/evaluate/save/load`) — **mesmo contrato** para churn etc.

### 7. `src/ml_pdv/training/train.py`

Treina o zoo, compara MAE na validação, escolhe o melhor.

### 8. `src/ml_pdv/training/pipeline.py`

Orquestra tudo + CLI `train-demand`.

## O que ainda NÃO está (de propósito)

- MLflow tracking (próximo passo natural: logar a tabela de métricas)
- Ingestão para o ML Postgres (`raw/clean`) — hoje lê direto do dump restaurado
- Promoção automática para a API (hoje a API ainda responde 503 até ligar o registry)

## Checklist se for criar outro problema (ex.: churn)

1. Novo `dataset.py` (ou módulo) com grain + target
2. Features sem leakage
3. Split adequado (temporal ou por cliente no tempo)
4. Baseline + modelos via `BaseForecastModel`
5. Métricas do domínio
6. Pipeline/CLI separado (`train-churn`)
