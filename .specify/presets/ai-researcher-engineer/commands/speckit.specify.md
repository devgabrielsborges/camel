---
description: "Create an AI/ML feature specification with research backing and engineering rigor"
---

## User Input

```text
$ARGUMENTS
```

You are creating a feature specification for an AI/ML system. Follow this workflow:

### Step 1: Understand the Feature

Parse the user's description to identify:
- The core problem being solved
- The target user or system consumer
- Any constraints mentioned (latency, accuracy, data, cost)

### Step 2: Research Context

Before writing the spec, gather relevant context:
- Search the codebase for related components, models, or pipelines
- Identify existing patterns, abstractions, and conventions
- Check for prior experiments or specs on similar topics (`specs/` directory)
- Note any dependencies or integration points

### Step 3: Read the Spec Template

Read `templates/spec-template.md` to see all sections that need to be filled.

### Step 4: Write the Specification

Create the spec file and fill every section thoughtfully:

1. **Problem Statement**: Ground it in evidence — user pain points, performance gaps, or business needs. Avoid vague motivation.

2. **Research Context**: Reference relevant papers, methods, or internal experiments. If the user mentioned a technique, cite foundational work.

3. **Proposed Solution**: Describe the approach with enough technical depth that an engineer could evaluate it. Include alternatives considered.

4. **Requirements**: Separate functional from non-functional. Every requirement must be testable. Include latency, throughput, accuracy targets.

5. **Metrics & Evaluation**: Define offline metrics (dev-time) and online metrics (production). Include guardrail metrics that must not regress.

6. **Data Requirements**: Specify training data, evaluation data, and data quality checks. Address data lineage and drift.

7. **Risks**: Be honest about what could go wrong. Every risk needs a mitigation.

8. **Rollout Plan**: Progressive rollout with clear rollback triggers.

9. **Acceptance Criteria**: Concrete, measurable outcomes — not vague statements.

### Principles to Follow

- **Precision over generality**: Specific numbers, thresholds, and examples.
- **Clean code alignment**: Solution should respect SOLID principles, separation of concerns, and testability.
- **Statistical rigor**: Evaluation plans must include proper methodology — significance tests, confidence intervals, baseline comparisons.
- **Production thinking**: Every spec must address how the feature operates at scale, fails gracefully, and gets monitored.
