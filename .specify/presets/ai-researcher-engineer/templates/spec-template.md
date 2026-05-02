# Feature Specification: [FEATURE NAME]

**Created**: [DATE]
**Status**: Draft | In Review | Approved | Implemented
**Author**: [NAME]
**Spec ID**: [PROJECT]-[NUMBER]

---

## 1. Problem Statement

Describe the problem this feature solves. Ground it in observable evidence — user pain points, system limitations, performance gaps, or business needs.

- **Who** is affected?
- **What** is the current behavior vs. desired behavior?
- **Why** does this matter now?

## 2. Research Context

Summarize relevant prior work — papers, internal experiments, existing approaches — that inform this solution.

| Source | Key Finding | Relevance |
|--------|-------------|-----------|
| [Paper / Experiment / System] | [Finding] | [How it applies] |

## 3. Proposed Solution

### 3.1 Approach

High-level description of the solution. Include the core algorithm, model architecture, or statistical method being used and why it was chosen over alternatives.

### 3.2 Alternatives Considered

| Alternative | Pros | Cons | Reason for Rejection |
|-------------|------|------|----------------------|
| [Alt 1] | | | |
| [Alt 2] | | | |

### 3.3 Technical Design

Key components, data flow, and module boundaries. Reference the system-design template for full architecture docs if needed.

```
[Component diagram or data flow sketch]
```

## 4. Requirements

### Functional Requirements

- [ ] FR-1: [Requirement]
- [ ] FR-2: [Requirement]

### Non-Functional Requirements

- [ ] NFR-1: Latency — [target, e.g. p99 < 200ms]
- [ ] NFR-2: Throughput — [target, e.g. 1000 req/s]
- [ ] NFR-3: Accuracy — [target metric and threshold]
- [ ] NFR-4: Data privacy — [constraints]

## 5. Metrics & Evaluation

Define how success is measured. Separate offline (development-time) from online (production) metrics.

### Offline Metrics

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|--------------------|
| [e.g. F1-score] | [current] | [goal] | [eval dataset / method] |

### Online Metrics

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|--------------------|
| [e.g. conversion rate] | [current] | [goal] | [A/B test / logging] |

### Guardrail Metrics

Metrics that must **not** degrade as a result of this change.

- [ ] [Metric] stays above [threshold]

## 6. Data Requirements

- **Training data**: [source, size, labeling strategy]
- **Evaluation data**: [holdout strategy, distribution]
- **Data pipeline**: [ingestion, transformation, storage]
- **Data quality checks**: [validation rules, drift detection]

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [Risk 1] | High/Med/Low | High/Med/Low | [Plan] |

## 8. Dependencies

- [ ] [Upstream service / dataset / team]
- [ ] [Infrastructure requirement]
- [ ] [Third-party API or library]

## 9. Rollout Plan

- **Phase 1**: [Shadow mode / canary / internal dogfood]
- **Phase 2**: [Gradual rollout %]
- **Phase 3**: [GA]
- **Rollback trigger**: [condition that triggers automatic rollback]

## 10. Acceptance Criteria

- [ ] AC-1: [Criterion with measurable outcome]
- [ ] AC-2: [Criterion with measurable outcome]
- [ ] AC-3: All unit and integration tests pass
- [ ] AC-4: Code review approved (clean code checklist passed)
- [ ] AC-5: Documentation updated

## 11. Open Questions

- [ ] [Question 1]
- [ ] [Question 2]
