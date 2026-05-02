# Project Plan: [PROJECT NAME]

**Created**: [DATE]
**Lead**: [NAME]
**Status**: Planning | Active | On Hold | Complete
**Related Spec**: [SPEC-ID or N/A]

---

## 1. Objective

One sentence describing what this project delivers and why it matters.

## 2. Background

Brief context: what problem exists, what prior work has been done, and what gap this project fills.

## 3. Success Criteria

| Criterion | Metric | Target | How Measured |
|-----------|--------|--------|-------------|
| [e.g. Model accuracy] | [F1-score] | [> 0.90] | [Eval dataset v2] |
| [e.g. Latency] | [p99 response time] | [< 200ms] | [Production monitoring] |
| [e.g. Adoption] | [% of users on new system] | [100%] | [Analytics dashboard] |

## 4. Scope

### In Scope

- [Deliverable 1]
- [Deliverable 2]

### Out of Scope

- [Explicitly excluded item 1]
- [Explicitly excluded item 2]

## 5. Approach

High-level strategy: how will the team get from current state to the target?

### 5.1 Technical Approach

- **Architecture**: [Reference system-design doc if applicable]
- **Key algorithms / methods**: [What and why]
- **Technology choices**: [Stack decisions and rationale]

### 5.2 Research Phase

- **Questions to answer**: [Before committing to implementation]
- **Experiments needed**: [Reference experiment template]
- **Decision gates**: [What must be true to proceed]

## 6. Phases & Milestones

### Phase 1: Research & Exploration — [START → END]

| Task | Owner | Estimate | Status | Depends On |
|------|-------|----------|--------|------------|
| Literature review on [topic] | [Name] | [X days] | Not Started | — |
| Baseline experiment | [Name] | [X days] | Not Started | Literature review |
| Data audit | [Name] | [X days] | Not Started | — |

**Exit criteria**: [What must be true to move to Phase 2]

### Phase 2: Development — [START → END]

| Task | Owner | Estimate | Status | Depends On |
|------|-------|----------|--------|------------|
| Data pipeline implementation | [Name] | [X days] | Not Started | Data audit |
| Model v1 training | [Name] | [X days] | Not Started | Data pipeline |
| API integration | [Name] | [X days] | Not Started | Model v1 |
| Unit & integration tests | [Name] | [X days] | Not Started | API integration |
| Code review | [Name] | [X days] | Not Started | Tests |

**Exit criteria**: [What must be true to move to Phase 3]

### Phase 3: Validation & Rollout — [START → END]

| Task | Owner | Estimate | Status | Depends On |
|------|-------|----------|--------|------------|
| A/B test or shadow deployment | [Name] | [X days] | Not Started | Code review |
| Performance benchmarking | [Name] | [X days] | Not Started | Shadow deployment |
| Monitoring & alerting setup | [Name] | [X days] | Not Started | Benchmarking |
| Gradual rollout | [Name] | [X days] | Not Started | Monitoring |
| Documentation | [Name] | [X days] | Not Started | Rollout |

**Exit criteria**: [Success criteria met, documentation complete]

## 7. Risks

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| [Risk 1] | High/Med/Low | High/Med/Low | [Plan] | [Name] |
| [Risk 2] | High/Med/Low | High/Med/Low | [Plan] | [Name] |

## 8. Dependencies

| Dependency | Type | Owner | Status | Impact if Delayed |
|-----------|------|-------|--------|-------------------|
| [Dep 1] | [Team / Infra / Data / External] | [Name/Team] | [Status] | [Impact] |

## 9. Resources

| Role | Person | Allocation | Duration |
|------|--------|------------|----------|
| [ML Engineer] | [Name] | [100%] | [Phases 1–3] |
| [Data Scientist] | [Name] | [50%] | [Phase 1] |
| [DevOps / Platform] | [Name] | [25%] | [Phases 2–3] |

### Infrastructure

| Resource | Specification | Cost | Duration |
|----------|-------------- |------|----------|
| [GPU cluster] | [4× A100 80GB] | [$X/mo] | [Phase 2] |
| [Storage] | [10 TB S3] | [$X/mo] | [Ongoing] |

## 10. Communication

- **Standup**: [Frequency, channel]
- **Progress report**: [Frequency, audience]
- **Decision log**: [Where architectural decisions are recorded]
- **Escalation path**: [Who to contact when blocked]

## 11. Definition of Done

- [ ] All success criteria met
- [ ] Code reviewed and merged
- [ ] Tests passing (unit, integration, model smoke tests)
- [ ] Monitoring and alerting operational
- [ ] Documentation complete (code docs, runbooks, user guide)
- [ ] Knowledge transfer completed
- [ ] Retrospective held
