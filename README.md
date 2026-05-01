# CAMEL — AI Agent Evaluation Pipeline

End-to-end pipeline for evaluating Q&A chatbot agents using the
[WeniEval Benchmark 2.0](https://huggingface.co/datasets/Weni/WeniEval-Benchmark-2.0.0).
Runs batch inference via the OpenAI Agents SDK, scores responses with
deterministic metrics and LLM-as-judge, and tracks everything in MLflow.

## Architecture

```
src/camel/
├── domain/          # Entities, value objects, domain services
├── application/     # Use cases, ports, DTOs
└── infrastructure/  # Adapters (OpenAI, MLflow, DuckDB), CLI, config
```

Clean Architecture with strict layer boundaries.
Domain-Driven Design with Evaluation, Session, and Trace entities.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose
- OpenAI API key

## Setup

```bash
git clone <repo-url> camel && cd camel
uv sync
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
docker compose up -d
```

## Data Preparation

```bash
camel prepare
```

Downloads the dataset and runs dbt bronze + silver transformations.

After a pipeline run, build gold analytical models:

```bash
camel prepare --skip-download --gold
```

## Usage

### Full Pipeline

```bash
camel run --limit 100
```

Runs inference, evaluation, and export in sequence.
Results written to `$RESULTS_DIR/predictions.csv`.

### Individual Commands

```bash
camel infer --limit 50
camel evaluate --run-id <mlflow-run-id>
camel export --run-id <mlflow-run-id>
```

### CLI Reference

```bash
camel --help
camel run --help
camel infer --help
camel evaluate --help
camel export --help
camel prepare --help
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--limit` | Max rows to process | all |
| `--batch-size` | Rows per batch | `$BATCH_SIZE` / 50 |
| `--concurrency` | Max concurrent calls | `$CONCURRENCY` / 10 |
| `--no-llm-judge` | Skip LLM scorers | false |
| `--experiment` | MLflow experiment name | WeniEval |
| `--output` | CSV output path | `$RESULTS_DIR`/predictions.csv |

## Scorers

| Scorer | Type | Description |
|--------|------|-------------|
| `token_overlap_f1` | Deterministic | Unigram F1 between response and content (tiktoken) |
| `class_exact_match` | Deterministic | Agent's class vs ground-truth `chosen_class_id` |
| `refusal_detection` | Deterministic | Detects refusal patterns (EN/PT/ES) via NLTK stemming |
| `correctness` | LLM Judge | MLflow Correctness scorer against expected response |
| `guidelines` | LLM Judge | MLflow Guidelines scorer against instructions |

## Data Pipeline

Medallion architecture with dbt + DuckDB:

- **Bronze**: Raw parquet ingestion (`stg_raw_dataset`)
- **Silver**: Filtered to `data_category_QA ∈ {positivo, negativo}` (`int_filtered_dataset`)
- **Gold**: Joined inference + evaluation results (`fct_inference_results`, `fct_evaluation_scores`)

## Testing

```bash
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest --cov=camel
```

## Code Quality

```bash
uv run pre-commit run --all-files
```

Checks: black, isort, autoflake, mypy (strict), vulture.

## Environment Variables

All configuration via `.env`. See `.env.example` for the full list.
Key variables: `OPENAI_API_KEY`, `OPENAI_MODEL`, `JUDGE_MODEL`,
`MLFLOW_TRACKING_URI`, `BATCH_SIZE`, `CONCURRENCY`.

## Output CSV

Columns: `id, question, prediction, data_category_QA, language, model,
correctness_score, guidelines_score, token_overlap_f1, class_exact_match,
refusal_detection`
