# Quickstart: WeniEval Pipeline

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed
- Docker + Docker Compose
- OpenAI API key with access to `gpt-4o-mini` (or chosen model)

## 1. Clone & Install

```bash
git clone <repo-url> camel && cd camel
uv sync
```

## 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

All other defaults work out of the box for local development.

## 3. Start Infrastructure

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** (port 5462) — MLflow backend store
- **MinIO** (port 9000/9001) — artifact storage
- **MLflow** (port 5000) — experiment tracking UI

Verify MLflow is healthy:

```bash
curl http://localhost:5000/health
```

## 4. Prepare Data

Ensure the dataset is at `data/raw/train.parquet`. If not present, download:

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('Weni/WeniEval-Benchmark-2.0.0', split='train')
ds.to_parquet('data/raw/train.parquet')
"
```

## 5. Run dbt Transformations

```bash
cd dbt && uv run dbt run && cd ..
```

This creates the bronze (raw) and silver (filtered + validated) tables in
`data/camel.duckdb`.

## 6. Run the Full Pipeline

```bash
uv run camel run --limit 100
```

This executes:
1. **Inference**: Processes 100 rows through the agent, stores traces in MLflow
2. **Evaluation**: Scores traces with deterministic + LLM-as-judge scorers
3. **Export**: Writes results to `results/predictions.csv`

## 7. View Results

### MLflow UI

Open http://localhost:5000 and navigate to experiment `WeniEval`.

### CSV Output

```bash
head results/predictions.csv
```

## Individual Commands

Run phases independently:

```bash
# Inference only (stores traces)
uv run camel infer --limit 50 --batch-size 25

# Evaluate existing traces (requires --run-id from inference output)
uv run camel evaluate --run-id <mlflow-run-id>

# Export to CSV (requires --run-id from inference output)
uv run camel export --run-id <mlflow-run-id> --output results/my_run.csv
```

## Full Dataset Run

Remove `--limit` to process all 3,887 rows:

```bash
uv run camel run
```

Estimated time: ~30 minutes (depends on API rate limits).
Estimated cost: ~$2-5 for inference + ~$1-3 for LLM judges (gpt-4o-mini).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MLflow unhealthy | Wait 30s after `docker compose up`; check `docker compose logs mlflow` |
| OpenAI rate limit | Reduce concurrency: `--concurrency 5` |
| OOM | Reduce batch size: `--batch-size 20` |
| MinIO bucket missing | Run `docker compose restart minio-init` |
| dbt errors | Ensure `data/raw/train.parquet` exists before `dbt run` |
