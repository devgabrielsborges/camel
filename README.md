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

Downloads the dataset to `data/bronze/`, applies stratified sampling to produce `data/silver/`, and runs dbt transformations.

Options:

```bash
camel prepare --skip-download          # reuse existing bronze parquet
camel prepare --skip-sample            # reuse existing silver parquet
camel prepare --skip-download --gold   # build gold models after a pipeline run
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
| `--model` / `-m` | Model name override | `$OPENAI_MODEL` |
| `--limit` | Max rows to process | all |
| `--batch-size` | Rows per batch | `$BATCH_SIZE` / 50 |
| `--concurrency` | Max concurrent calls | `$CONCURRENCY` / 10 |
| `--no-llm-judge` | Skip LLM scorers | false |
| `--experiment` | MLflow experiment name | WeniEval |
| `--output` | CSV output path | `$RESULTS_DIR`/predictions.csv |

### LiteLLM Support

Set `LLM_PROVIDER=litellm` in `.env` to use any provider supported by
[LiteLLM](https://docs.litellm.ai/docs/providers). The `--model` flag
(or `OPENAI_MODEL` env var) uses the `provider/model` format:

```bash
# Anthropic
camel run --model anthropic/claude-3-haiku-20240307 --limit 10

# Azure OpenAI
camel run --model azure/gpt-4o --limit 10

# Groq
camel infer --model groq/llama3-8b-8192 --limit 10
```

When using `LLM_PROVIDER=openai` (default), the model name is passed directly
to the OpenAI Agents SDK.

## Scorers

| Scorer | Type | Description |
|--------|------|-------------|
| `token_overlap_f1` | Deterministic | Unigram F1 between response and content (tiktoken) |
| `class_exact_match` | Deterministic | Agent's class vs ground-truth `chosen_class_id` |
| `refusal_detection` | Deterministic | Detects refusal patterns (EN/PT/ES) via NLTK stemming |
| `groundedness` | TruLens | Response grounded in source content (CoT reasoning) |
| `pass@3` | Composite | At least 1 of 3 responses (temp=0.7) passes threshold |
| `failure_mode` | Derived | Categorizes prediction into failure type |
| `correctness` | LLM Judge | MLflow Correctness scorer against expected response |
| `guidelines` | LLM Judge | MLflow Guidelines scorer against instructions |

### Failure Modes

Each prediction is classified into one of:

| Mode | Condition |
|------|-----------|
| `correct_extraction` | High overlap + positivo category |
| `correct_refusal` | Refusal detected + negativo category |
| `false_refusal` | Refusal detected + positivo category |
| `hallucination` | Moderate overlap + negativo category |
| `off_topic` | Very low overlap + no refusal |
| `partial_answer` | All other cases |

### Capability Verdict

After scoring, the pipeline produces a verdict: **capable**, **not_capable**, or **inconclusive**.

Criteria:
1. **Positivo**: mean `token_overlap_f1` >= threshold (model can extract answers)
2. **Negativo**: refusal rate >= threshold OR low overlap (model avoids hallucination)
3. **Discrimination**: significant delta between positivo/negativo overlap (model differentiates categories)

## Data Pipeline

Medallion architecture with dbt + DuckDB:

```
data/
├── bronze/   # Raw dataset from HuggingFace
├── silver/   # Stratified sample (filtered + sampled)
└── camel.duckdb
```

- **Bronze** (`data/bronze/`): Raw parquet ingestion (`stg_raw_dataset`)
- **Silver** (`data/silver/`): Stratified sampling preserving class proportions (`data_category_QA ∈ {positivo, negativo}`), loaded into dbt as `int_filtered_dataset`
- **Gold** (dbt + `results/`): Joined inference + evaluation results (`fct_inference_results`, `fct_evaluation_scores`)

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

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_MODEL` | Model for inference | `gpt-4o-mini` |
| `JUDGE_MODEL` | Model for LLM-as-judge and groundedness | `gpt-4o-mini` |
| `LLM_PROVIDER` | `openai` or `litellm` | `openai` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `RAW_PARQUET_PATH` | Bronze layer path | `data/bronze/train.parquet` |
| `SILVER_PARQUET_PATH` | Silver layer path | `data/silver/train_sample.parquet` |
| `SAMPLE_FRACTION` | Fraction of dataset to sample | `0.1` |
| `SAMPLE_SEED` | Random seed for reproducibility | `42` |
| `PASS_AT_K` | Number of responses per question for Pass@k | `3` |
| `PASS_AT_K_TEMPERATURE` | Temperature for diverse Pass@k responses | `0.7` |
| `BATCH_SIZE` | Rows per batch | `50` |
| `CONCURRENCY` | Max concurrent calls | `10` |

## Output JSONL

Results are exported as JSONL (one JSON object per line) to `results/predictions.jsonl`.
New runs **append** to the file, so historical results are preserved. Each record includes
`run_id` and `timestamp` fields to distinguish runs.

Fields: `id, run_id, timestamp, question, prediction, data_category_QA, language, model,
correctness_score, guidelines_score, token_overlap_f1, class_exact_match,
refusal_detection, groundedness_score, pass_at_k, pass_at_k_best_score,
failure_mode`
