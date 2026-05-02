# Tasks: Pass@k + Groundedness Evaluation

**Plan**: `docs/plan.md` | **Spec**: `docs/spec-evaluation-methodology.md`

## Phase 1: Setup

- [x] T001 Add `trulens-providers-litellm` dependency via `uv add` (trulens-providers-openai incompatible with openai-agents)
- [x] T002 Add `pass_at_k`, `pass_at_k_temperature` settings to `src/camel/infrastructure/config/settings.py`
- [x] T003 Add `PASS_AT_K=3` and `PASS_AT_K_TEMPERATURE=0.7` to `.env` and `.env.example`

## Phase 2: Foundational — Domain Layer

- [x] T004 [P] Create `PassAtKResult` value object in `src/camel/domain/value_objects/pass_at_k_result.py`
- [x] T005 [P] Implement `pass_at_k()` domain service function in `src/camel/domain/services/scoring.py`
- [x] T006 [P] Create `GroundednessPort` abstract interface in `src/camel/application/ports/groundedness_port.py`
- [x] T007 Create unit tests for pass_at_k logic in `tests/unit/domain/test_pass_at_k.py`

## Phase 3: User Story 1 — Capability Evaluation with Pass@k + Groundedness (P1)

- [ ] T008 [US1] Implement `TruLensGroundednessAdapter` in `src/camel/infrastructure/adapters/trulens_groundedness.py`
- [ ] T009 [US1] Add groundedness provider factory function in `src/camel/infrastructure/factories/scorer_factory.py`
- [ ] T010 [US1] Add groundedness MLflow scorer wrapper in `src/camel/infrastructure/adapters/mlflow_scorer.py`
- [ ] T011 [US1] Modify `RunInference` to generate k=3 responses per question in `src/camel/application/use_cases/run_inference.py`
- [ ] T012 [US1] Modify `RunEvaluation` to compute groundedness + pass@k in `src/camel/application/use_cases/run_evaluation.py`
- [ ] T013 [US1] Modify `ExportResults` to include groundedness_score, pass_at_k, pass_at_k_best_score columns in `src/camel/application/use_cases/export_results.py`
- [ ] T014 [US1] Update dbt gold model to include groundedness column in `dbt/models/gold/fct_evaluation_scores.sql`
- [ ] T015 [US1] Create unit test for TruLens adapter in `tests/unit/infrastructure/test_trulens_groundedness.py`

## Phase 4: User Story 2 — Explainability via Failure Mode Categorization (P2)

- [ ] T016 [US2] Implement failure mode categorization service in `src/camel/domain/services/failure_modes.py`
- [ ] T017 [US2] Integrate failure mode classification into evaluation output in `src/camel/application/use_cases/run_evaluation.py`
- [ ] T018 [US2] Add `failure_mode` column to CSV export in `src/camel/application/use_cases/export_results.py`
- [ ] T019 [US2] Add failure mode distribution to MLflow metrics logging in `src/camel/infrastructure/adapters/mlflow_tracker.py`

## Phase 5: User Story 3 — Report Generation (P3)

- [ ] T020 [US3] Add capability verdict logic (capable/not_capable/inconclusive) in `src/camel/domain/services/verdict.py`
- [ ] T021 [US3] Integrate verdict into pipeline output in `src/camel/application/use_cases/run_pipeline.py`
- [ ] T022 [US3] Log verdict + per-category breakdown to MLflow as final step in `src/camel/infrastructure/adapters/mlflow_tracker.py`

## Phase 6: Polish

- [ ] T023 Update README with new scorers (groundedness, pass@3) and verdict in `README.md`
- [ ] T024 Update `docs/notes.md` with final methodology description
- [ ] T025 Verify full pipeline run end-to-end with `camel prepare && camel run --limit 10`

## Dependencies

```text
T001 → T002 → T003 (sequential setup)
T004, T005, T006 can run in parallel (no shared files)
T007 depends on T004 + T005
T008 depends on T006
T009 depends on T008
T010 depends on T009
T011 depends on T002 (needs pass_at_k settings)
T012 depends on T005 + T010 + T011
T013 depends on T012
T014 depends on T013
T015 depends on T008
T016 depends on T005 (uses scorer results)
T017 depends on T016
T018 depends on T017
T019 depends on T017
T020 depends on T012 (needs all scores computed)
T021 depends on T020
T022 depends on T021
T023 depends on T022
T024 depends on T023
T025 depends on all above
```

## Parallel Execution Opportunities

### Phase 2 parallelism:
- T004, T005, T006 can all execute simultaneously (different files, no dependencies)

### Phase 3 parallelism:
- T008 and T011 can execute simultaneously (different files)
- T015 can execute as soon as T008 is done (independent of T009-T014)

### Phase 4 + Phase 5 partial overlap:
- T016 can start as soon as Phase 3's T012 completes
- T020 can also start after T012 completes
- T016 and T020 can run in parallel (different files)

## Implementation Strategy

**MVP (User Story 1 only)**: Tasks T001–T015 produce a working pipeline with Pass@k and groundedness scoring. This alone satisfies FR-008, FR-009, FR-010, FR-011 and provides the core evaluation enhancement.

**Incremental delivery**:
1. Phase 1–3: Core scoring (Pass@k + groundedness) — delivers measurable value
2. Phase 4: Explainability layer — enriches conclusions
3. Phase 5: Verdict + report — wraps everything into a conclusion
4. Phase 6: Documentation and validation

**Total tasks**: 25
- Setup: 3
- Foundational: 4
- US1 (P1): 8
- US2 (P2): 4
- US3 (P3): 3
- Polish: 3
