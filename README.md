<p align="center">
  <img src="docs/camelbg.jpeg" alt="CAMEL" width="160"/>
</p>

<h1 align="center">CAMEL</h1>
<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version">
</p>
<p align="center"><strong>Capability Assessment Methodology for Evaluating Large Language Models</strong></p>

<p align="center">
  End-to-end pipeline for evaluating Q&A chatbot agents using the
  <a href="https://huggingface.co/datasets/Weni/WeniEval-Benchmark-2.0.0">WeniEval Benchmark 2.0</a>.<br/>
  Runs batch inference via the OpenAI Agents SDK, scores responses with
  deterministic xAI metrics and LLM-as-judge, and produces a statistically grounded capability verdict tracked in MLflow.
</p>

---

## Architecture

```
src/camel/
├── domain/          # Entities, value objects, domain services
├── application/     # Use cases, ports, DTOs
└── infrastructure/  # Adapters (OpenAI, MLflow, DuckDB), CLI, config, dashboard
```

Built on **Clean Architecture** with strict layer boundaries and **Domain-Driven Design** with Evaluation, Session, and Trace entities.

**Design patterns**: Ports & Adapters (swap OpenAI/LiteLLM without touching use cases), Factory (agent instantiation from config strings), Decorator (CachedAgent for disk persistence and Pass@k).

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

## Data Pipeline

Medallion architecture with dbt + DuckDB:

```
data/
├── bronze/   # Raw dataset from HuggingFace
├── silver/   # Stratified sample (filtered + sampled)
└── camel.duckdb
```

| Layer | Source | Description |
|-------|--------|-------------|
| **Bronze** | HuggingFace | Raw parquet ingestion (`stg_raw_dataset`) |
| **Silver** | Bronze | Stratified sampling with inverse-frequency weights, preserving class proportions (`data_category_QA ∈ {positivo, negativo}`) |
| **Gold** | dbt + results | Joined inference + evaluation results (`fct_inference_results`, `fct_evaluation_scores`) |

### Prepare Data

```bash
camel prepare
```

Downloads dataset to `data/bronze/`, applies stratified sampling to `data/silver/`, and runs dbt transformations.

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

Runs inference, evaluation, and export in sequence. Results appended to `$RESULTS_DIR/predictions.jsonl`.

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
camel dashboard --help
camel derive-thresholds --help
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` / `-m` | Model name override | `$OPENAI_MODEL` |
| `--limit` | Max rows to process | all |
| `--batch-size` | Rows per batch | `$BATCH_SIZE` / 50 |
| `--concurrency` | Max concurrent calls | `$CONCURRENCY` / 10 |
| `--no-llm-judge` | Skip LLM scorers | false |
| `--legacy-verdict` | Skip statistical verdict (legacy only) | false |
| `--threshold-profile` | Path to ThresholdProfile JSON | `$THRESHOLD_PROFILE_PATH` |
| `--experiment` | MLflow experiment name | WeniEval |
| `--output` | JSONL output path | `$RESULTS_DIR`/predictions.jsonl |

### LiteLLM Support

Set `LLM_PROVIDER=litellm` in `.env` to use any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers). The `--model` flag uses the `provider/model` format:

```bash
camel run --model anthropic/claude-3-haiku-20240307 --limit 10
camel run --model azure/gpt-4o --limit 10
camel infer --model groq/llama3-8b-8192 --limit 10
```

When using `LLM_PROVIDER=openai` (default), the model name is passed directly to the OpenAI Agents SDK.

## Scorers

### Deterministic xAI Metrics

| Scorer | Description |
|--------|-------------|
| `token_overlap_f1` | Unigram F1 between response and content (tiktoken `o200k_base` encoding) |
| `rouge_l` | LCS-based F1 over tiktoken token ID sequences via dynamic programming |
| `self_consistency` | Average pairwise token overlap F1 across k=3 responses (deterministic semantic entropy approximation) |
| `chunk_attribution` | Per-chunk F1 + Shannon entropy + Spearman correlation with chunk relevance scores |
| `hedging_detection` | Multilingual pattern matching (EN/PT/ES) for uncertainty language |
| `question_response_overlap` | Unigram F1 between question and response tokens (off-topic detection) |
| `response_length_ratio` | Token count ratio of response to reference content (verbosity calibration) |
| `class_exact_match` | Agent's class vs ground-truth `chosen_class_id` |
| `refusal_detection` | Detects refusal patterns (EN/PT/ES) via NLTK stemming |
| `pass@3` | At least 1 of 3 responses (temp=0.7) passes F1 threshold |

### LLM-as-Judge

| Scorer | Description |
|--------|-------------|
| `correctness` | MLflow Correctness scorer against expected response |
| `guidelines` | MLflow Guidelines scorer against instructions |
| `groundedness` | TruLens — response grounded in source content (CoT reasoning) |

### Failure Modes

Each prediction is classified into one of seven modes via a hierarchical decision tree (first match wins):

| Mode | Condition | Outcome |
|------|-----------|---------|
| `false_refusal` | Refusal detected + positivo | Error |
| `correct_refusal` | Refusal detected + negativo | Correct |
| `correct_extraction` | Overlap > 0.5 + positivo | Correct |
| `hallucination` | Overlap > 0.3 + negativo | Error |
| `off_topic` | Overlap < 0.1 + no refusal | Error |
| `hedging_answer` | Hedging detected + positivo | Warning |
| `partial_answer` | Otherwise | Partial |

## Statistical Verdict

The capability verdict is determined through a two-gate decision mechanism requiring both statistical and practical significance.

### Threshold Derivation

Bootstrap resampling (B=10,000) from reference model runs establishes percentile-based thresholds with 95% CIs:

```bash
camel derive-thresholds --models gpt-5.4-mini --models gpt-5.4 --models gpt-5.1
```

For composite metrics (discrimination delta), the bootstrap operates over paired category means: `δ* = |mean(positivo) - mean(negativo)|`.

### Hypothesis Testing

The pipeline automatically routes each metric to the appropriate test:

| Metric Type | Paired Test | Unpaired Test | Effect Size |
|-------------|-------------|---------------|-------------|
| Continuous (`token_overlap_f1`, `rouge_l`, etc.) | Wilcoxon signed-rank | Mann-Whitney U | Cohen's d |
| Binary (`refusal_detection`, `pass_at_k`) | McNemar's exact | Chi-squared | Odds ratio |
| Composite (`discrimination_delta`) | Bootstrap CI | Bootstrap CI | Relative diff |

Pairing is detected via session ID overlap (≥80% common IDs). McNemar uses exact binomial when discordant pairs < 25.

### Multiple Testing Correction

Benjamini-Hochberg FDR correction at α = 0.05. Bootstrap results are excluded from the FDR pool (CI-based rejection, not p-value-based).

### Verdict Decision

A **critical failure** requires both gates triggered for a metric in the critical set `{(F1, positivo), (F1, negativo), (refusal, negativo), (Δ_disc, global)}`:

| Verdict | Condition |
|---------|-----------|
| **CAPABLE** | No critical failures, no inconclusive signals |
| **NOT_CAPABLE** | At least one critical failure (reject + medium/large effect) |
| **INCONCLUSIVE** | No critical failures, but rejection with small effect size |

Effect size thresholds (Cohen's conventions):
- Continuous: small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8
- Binary (odds ratio): small ≥ 1.5, medium ≥ 2.0, large ≥ 3.0

When no `ThresholdProfile` exists, the pipeline falls back to legacy verdict with a warning.

## Dashboard

Interactive Streamlit dashboard for exploring evaluation results.

```bash
camel dashboard
camel dashboard --db-path data/gold/camel.duckdb --port 8502 --no-browser
```

| Tab | Description |
|-----|-------------|
| **Overview** | KPI cards (mean ± std) and radar chart for all metrics |
| **Comparison** | Descriptive statistics; pairwise model comparison with Welch's t-test and significance highlighting |
| **Distributions** | Side-by-side box plots per metric, colored by model |
| **Failure Modes** | Stacked bar charts and Sankey diagram (category → refusal → failure mode) |
| **Deep Dive** | Cost-performance scatter, performance-vs-complexity charts, and session inspector |

## MLflow Tracking

Every evaluation run is logged with full lineage:

- **Run metadata**: model, prompt version, dataset registration, verdict tags
- **Inference tracing**: OpenAI autolog + LiteLLM callback for provider-agnostic spans
- **Metrics**: per-metric aggregates, failure mode counts/rates, per-test p-values, effect sizes
- **Prompt registry**: version-tracked via MLflow GenAI API
- **Dataset registry**: records merged into MLflow datasets for provenance

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
| `PASS_AT_K` | Number of responses per question | `3` |
| `PASS_AT_K_TEMPERATURE` | Temperature for diverse responses | `0.7` |
| `BATCH_SIZE` | Rows per batch | `50` |
| `CONCURRENCY` | Max concurrent calls | `10` |
| `DUCKDB_PATH` | Path to DuckDB database | `data/gold/camel.duckdb` |
| `THRESHOLD_PROFILE_PATH` | ThresholdProfile JSON for statistical verdict | `data/thresholds/profile.json` |

## Output

Results are exported as JSONL (one JSON object per line) to `results/predictions.jsonl`. New runs **append** to preserve historical results. Each record includes `run_id` and `timestamp` to distinguish runs.

## Links

- [GitHub](https://github.com/devgabrielsborges/camel)
- [Codeberg](https://codeberg.org/devgabrielsborges/camel)
- [WeniEval Benchmark 2.0 Dataset](https://huggingface.co/datasets/Weni/WeniEval-Benchmark-2.0.0)
