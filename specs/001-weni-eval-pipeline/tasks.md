# Tasks: WeniEval — AI Agent Evaluation Pipeline

**Input**: Design documents from `/specs/001-weni-eval-pipeline/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Included per Constitution Principle VI (Test-First Development).

**Organization**: Tasks grouped by user story for independent implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths included in all descriptions

## Path Conventions

- Single project: `src/camel/` for source, `tests/` at repository root
- dbt models: `dbt/` at repository root
- Infrastructure: repo root (`docker-compose.yaml`, `.env.example`, `Dockerfile`)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project structure, tooling, Docker services, dbt pipeline

- [X] T001 Create package directory structure per plan.md in src/camel/{domain,application,infrastructure}/
- [X] T002 [P] Update .env.example with all required variables per research.md §8 in .env.example
- [X] T003 [P] Fix docker-compose.yaml MLflow command syntax (missing backslash) and set MLFLOW_S3_ENDPOINT_URL to internal http://minio:9000 in docker-compose.yaml
- [X] T004 [P] Create Dockerfile for eval service with uv + Python 3.12 in Dockerfile
- [X] T005 [P] Configure pyproject.toml: fix isort known_first_party to "camel", fix vulture paths to "src/camel" in pyproject.toml
- [X] T006 Initialize dbt project with DuckDB profile in dbt/dbt_project.yml and dbt/profiles.yml
- [X] T007 [P] Create bronze model: raw parquet ingestion in dbt/models/bronze/stg_raw_dataset.sql
- [X] T008 Create silver model: filter data_category_QA ∈ {positivo, negativo} + validate in dbt/models/silver/int_filtered_dataset.sql
- [X] T009 Create Pydantic BaseSettings config module in src/camel/infrastructure/config/settings.py
- [X] T010 [P] Create results/ directory with .gitkeep in results/.gitkeep

**Checkpoint**: `docker compose up -d` runs; `dbt run` produces silver table; settings load from .env

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Domain layer + application ports that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T011 [P] Create DatasetRecord value object in src/camel/domain/value_objects/dataset_record.py
- [X] T012 [P] Create ModelConfig value object in src/camel/domain/value_objects/model_config.py
- [X] T013 [P] Create Score value object in src/camel/domain/value_objects/score.py
- [X] T014 [P] Create PromptTemplate value object in src/camel/domain/value_objects/prompt_template.py
- [X] T015 [P] Create TokenUsage, Chunk, ClassDef, ToolCall value objects in src/camel/domain/value_objects/__init__.py
- [X] T016 Create Trace entity with invariants in src/camel/domain/entities/trace.py
- [X] T017 Create Session entity with invariants in src/camel/domain/entities/session.py
- [X] T018 Create Evaluation aggregate root with status transitions in src/camel/domain/entities/evaluation.py
- [X] T019 [P] Create DatasetRepository protocol in src/camel/domain/repositories/dataset_repository.py
- [X] T020 [P] Create TraceRepository protocol in src/camel/domain/repositories/trace_repository.py
- [X] T021 [P] Create EvaluationRepository protocol in src/camel/domain/repositories/evaluation_repository.py
- [X] T022 [P] Create AgentPort protocol in src/camel/application/ports/agent_port.py
- [X] T023 [P] Create TrackerPort protocol in src/camel/application/ports/tracker_port.py
- [X] T024 [P] Create ScorerPort protocol in src/camel/application/ports/scorer_port.py
- [X] T025 Create PromptRenderer domain service with Jinja2 + tiktoken in src/camel/domain/services/prompt_renderer.py
- [X] T026 [P] Create DTOs (InferenceRequest, InferenceResult, EvaluationResult) in src/camel/application/dtos/
- [X] T027 Create Typer app root with global --verbose flag in src/camel/infrastructure/cli/app.py

**Checkpoint**: Foundation ready — domain types + ports defined; user story implementation can begin

---

## Phase 3: User Story 1 — Batch Inference (Priority: P1) 🎯 MVP

**Goal**: Run the model-under-test on the filtered dataset, produce traces grouped by session in MLflow

**Independent Test**: `camel infer --limit 2` produces 2 traces in MLflow with valid session linkage

### Tests for User Story 1

- [X] T028 [P] [US1] Unit test for PromptRenderer in tests/unit/domain/test_prompt_renderer.py
- [X] T029 [P] [US1] Unit test for RunInference use case (mocked ports) in tests/unit/application/test_run_inference.py
- [X] T030 [P] [US1] Integration test for DuckDB dataset adapter in tests/integration/test_duckdb_dataset.py
- [X] T031 [P] [US1] Integration test for OpenAI agent adapter in tests/integration/test_openai_agent.py
- [X] T031b [P] [US1] Integration test for MLflow prompt registration in tests/integration/test_mlflow_tracker.py

### Implementation for User Story 1

- [X] T032 [US1] Implement DuckDB dataset adapter (DatasetRepository) in src/camel/infrastructure/adapters/duckdb_dataset.py
- [X] T033 [US1] Implement OpenAI agent adapter (AgentPort) with batch + semaphore in src/camel/infrastructure/adapters/openai_agent.py
- [X] T034 [US1] Implement MLflow tracker adapter (TrackerPort) — experiment, run, traces, sessions, prompt registration in src/camel/infrastructure/adapters/mlflow_tracker.py
- [X] T035 [US1] Implement AgentFactory in src/camel/infrastructure/factories/agent_factory.py
- [X] T036 [US1] Implement DatasetFactory in src/camel/infrastructure/factories/dataset_factory.py
- [X] T037 [US1] Implement RunInference use case — orchestrate batch inference with memory management in src/camel/application/use_cases/run_inference.py
- [X] T038 [US1] Implement RegisterDataset use case — register MLflow evaluation dataset in src/camel/application/use_cases/register_dataset.py
- [X] T039 [US1] Implement `camel infer` CLI command with all options per contracts/cli.md in src/camel/infrastructure/cli/infer_cmd.py
- [X] T040 [US1] Create conftest.py with shared fixtures (settings, mock adapters) in tests/conftest.py

**Checkpoint**: `camel infer --limit 2` works end-to-end; traces visible in MLflow UI; session linkage verified

---

## Phase 4: User Story 2 — Evaluation & Scoring (Priority: P2)

**Goal**: Score all cached traces using deterministic scorers and LLM-as-judge without re-running inference

**Independent Test**: `camel evaluate --run-id <existing-run>` produces scores logged to MLflow run

### Tests for User Story 2

- [X] T041 [P] [US2] Unit test for scoring domain service (token_overlap_f1, class_exact_match, refusal_detection) in tests/unit/domain/test_scoring.py
- [X] T042 [P] [US2] Unit test for aggregation domain service in tests/unit/domain/test_aggregation.py
- [X] T043 [P] [US2] Unit test for RunEvaluation use case (mocked ports) in tests/unit/application/test_run_evaluation.py
- [X] T044 [P] [US2] Integration test for MLflow scorer adapter in tests/integration/test_mlflow_scorer.py

### Implementation for User Story 2

- [X] T045 [US2] Implement scoring domain service — token_overlap_f1, class_exact_match, refusal_detection in src/camel/domain/services/scoring.py
- [X] T046 [US2] Implement aggregation domain service — mean, std, per-category breakdown in src/camel/domain/services/aggregation.py
- [X] T047 [US2] Implement MLflow scorer adapter (ScorerPort) — Correctness + Guidelines judges in src/camel/infrastructure/adapters/mlflow_scorer.py
- [X] T048 [US2] Implement ScorerFactory in src/camel/infrastructure/factories/scorer_factory.py
- [X] T049 [US2] Implement RunEvaluation use case — retrieve traces, build eval dataset, run scorers, aggregate in src/camel/application/use_cases/run_evaluation.py
- [X] T050 [US2] Implement `camel evaluate` CLI command with all options per contracts/cli.md in src/camel/infrastructure/cli/evaluate_cmd.py

**Checkpoint**: `camel evaluate` scores traces from US1; deterministic + LLM judge scores logged to MLflow

---

## Phase 5: User Story 3 — Export & Full Pipeline (Priority: P3)

**Goal**: Export results to CSV and orchestrate the full pipeline via `camel run`

**Independent Test**: `camel run --limit 2` completes infer→evaluate→export; CSV at results/predictions.csv

### Tests for User Story 3

- [X] T051 [P] [US3] Unit test for ExportResults use case in tests/unit/application/test_export_results.py
- [X] T052 [P] [US3] Unit test for RunPipeline use case (verify orchestration + halt-on-failure) in tests/unit/application/test_run_pipeline.py

### Implementation for User Story 3

- [X] T053 [US3] Implement ExportResults use case — query traces + scores, write CSV in src/camel/application/use_cases/export_results.py
- [X] T054 [US3] Implement RunPipeline use case — orchestrate infer → evaluate → export with error propagation in src/camel/application/use_cases/run_pipeline.py
- [X] T055 [US3] Implement `camel export` CLI command per contracts/cli.md in src/camel/infrastructure/cli/export_cmd.py
- [X] T056 [US3] Implement `camel run` CLI command per contracts/cli.md in src/camel/infrastructure/cli/run_cmd.py
- [X] T057 [US3] Create gold dbt model: join inference results with evaluation scores in dbt/models/gold/fct_inference_results.sql
- [X] T058 [US3] Create gold dbt model: evaluation score aggregations in dbt/models/gold/fct_evaluation_scores.sql

**Checkpoint**: `camel run --limit 2` completes full pipeline; CSV exported; MLflow experiment fully populated

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation, full run, documentation

- [ ] T059 End-to-end test on 2-sample subset (full pipeline smoke test + tracemalloc peak < 8GB assertion) in tests/integration/test_e2e_pipeline.py
- [ ] T060 [P] Validate quickstart.md steps work from clean state
- [ ] T061 Full dataset run (all 3,887 rows) and generate report
- [ ] T062 [P] Verify all code passes pre-commit hooks (black, isort, autoflake, mypy, vulture)
- [ ] T063 [P] Update README.md with project overview, setup, and usage

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on T001 (package structure) — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Phase 2 completion
- **User Story 2 (Phase 4)**: Depends on Phase 2 completion; can run after US1 produces traces
- **User Story 3 (Phase 5)**: Depends on US1 + US2 (needs traces + scores to export)
- **Polish (Phase 6)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (Batch Inference)**: Can start after Phase 2 — no dependencies on US2/US3
- **US2 (Evaluation)**: Requires traces from US1 for integration testing; domain logic is independent
- **US3 (Export + Run)**: Requires both US1 and US2 complete (orchestrates both)

### Within Each User Story

- Tests written FIRST (Red phase)
- Value objects → Entities → Repositories
- Adapters → Factories
- Use cases → CLI commands
- Story complete → checkpoint validation

### Parallel Opportunities

- Phase 1: T002, T003, T004, T005, T007, T010 all parallelizable
- Phase 2: T011–T015 (all VOs), T019–T024 (all ports), T026 (DTOs) parallelizable
- Phase 3: T028–T031 (all tests) parallelizable; T035, T036 parallelizable
- Phase 4: T041–T044 (all tests) parallelizable
- Phase 5: T051, T052 (tests) parallelizable; T057, T058 (dbt models) parallelizable
- Phase 6: T060, T062, T063 parallelizable

---

## Parallel Example: Phase 2 (Foundational)

```bash
# Launch all value objects in parallel:
Task T011: "Create DatasetRecord VO in src/camel/domain/value_objects/dataset_record.py"
Task T012: "Create ModelConfig VO in src/camel/domain/value_objects/model_config.py"
Task T013: "Create Score VO in src/camel/domain/value_objects/score.py"
Task T014: "Create PromptTemplate VO in src/camel/domain/value_objects/prompt_template.py"
Task T015: "Create TokenUsage, Chunk, ClassDef, ToolCall VOs"

# Launch all ports in parallel:
Task T019: "Create DatasetRepository protocol"
Task T020: "Create TraceRepository protocol"
Task T021: "Create EvaluationRepository protocol"
Task T022: "Create AgentPort protocol"
Task T023: "Create TrackerPort protocol"
Task T024: "Create ScorerPort protocol"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (infra + dbt + config)
2. Complete Phase 2: Foundational (domain types + ports)
3. Complete Phase 3: User Story 1 (batch inference)
4. **STOP and VALIDATE**: `camel infer --limit 2` runs successfully
5. Traces visible in MLflow UI with session linkage

### Incremental Delivery

1. Setup + Foundational → Infrastructure validated
2. US1 (Inference) → Traces in MLflow (MVP!)
3. US2 (Evaluation) → Scores computed and logged
4. US3 (Export + Run) → Full pipeline one-command
5. Polish → Full dataset run + report

### Sequential Execution (solo developer)

1. Phase 1 → Phase 2 → Phase 3 → validate MVP
2. Phase 4 → validate scoring
3. Phase 5 → Phase 6 → final validation

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [US*] label maps task to its user story for traceability
- Each user story is independently testable at its checkpoint
- Commit after each task or logical group (never `git add .`)
- Stop at any checkpoint to validate independently
- Constitution: all code must pass mypy --strict; all commits use template + Signed-off-by
