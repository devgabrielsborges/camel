# Code Review: [PR / Change Title]

**Reviewer**: [NAME]
**Author**: [NAME]
**Date**: [DATE]
**PR / Commit**: [LINK]
**Verdict**: Approved | Request Changes | Needs Discussion

---

## 1. Summary

One-paragraph description of what this change does and why.

## 2. Clean Code Checklist

### 2.1 Naming & Readability

- [ ] Names reveal intent — variables, functions, and classes describe what they represent
- [ ] No abbreviations that require domain knowledge to decode
- [ ] Consistent naming conventions across the change
- [ ] Functions do one thing and their name says what that thing is
- [ ] No misleading names (e.g. `list` for something that isn't a list)

### 2.2 Functions & Methods

- [ ] Functions are small (< 20 lines as a guideline, not a rule)
- [ ] No more than 3 parameters (use objects/dataclasses for more)
- [ ] No side effects hidden in function names (e.g. `get_x()` should not modify state)
- [ ] Command-query separation respected where appropriate
- [ ] No flag arguments that change behavior — split into separate functions

### 2.3 Error Handling

- [ ] Exceptions used for exceptional cases, not control flow
- [ ] Errors are specific (no bare `except:` or `catch (Exception e)`)
- [ ] Error messages are actionable — they say what went wrong and suggest a fix
- [ ] Fail-fast principle: invalid state detected early
- [ ] Resource cleanup guaranteed (context managers, finally blocks)

### 2.4 Comments & Documentation

- [ ] Code is self-documenting — comments explain *why*, not *what*
- [ ] No commented-out code
- [ ] Public APIs have docstrings with types, parameters, return values, and examples
- [ ] Complex algorithms have references (paper, issue, design doc)
- [ ] TODO comments have an associated ticket or owner

### 2.5 Structure & Dependencies

- [ ] No circular dependencies
- [ ] Dependency injection over hard-coded dependencies
- [ ] Single Responsibility Principle — each module/class has one reason to change
- [ ] Open/Closed Principle — extensible without modifying existing code
- [ ] No god classes or god functions

## 3. ML / Data Science Checklist

### 3.1 Data Handling

- [ ] No data leakage between train/val/test splits
- [ ] Data preprocessing is deterministic and reproducible
- [ ] Feature transformations are fitted on train set only, then applied to val/test
- [ ] Data validation checks present (schema, ranges, nulls)
- [ ] Raw data is never mutated — transformations produce new artifacts

### 3.2 Model Code

- [ ] Random seeds set for reproducibility
- [ ] Hyperparameters externalized (config file, not hardcoded)
- [ ] Model architecture and training loop are separated
- [ ] Evaluation metrics match the business objective
- [ ] No silent numerical issues (overflow, underflow, NaN propagation)

### 3.3 Experiment Tracking

- [ ] Metrics, parameters, and artifacts logged to experiment tracker
- [ ] Model version and data version recorded together
- [ ] Training configuration is fully reconstructable from logs
- [ ] Comparison against established baseline documented

### 3.4 Pipeline & Deployment

- [ ] Inference code does not depend on training-only libraries
- [ ] Model serialization format is versioned and documented
- [ ] Pre/post-processing steps are identical between training and serving
- [ ] Batch and real-time paths produce identical results for the same input
- [ ] Graceful degradation if model is unavailable

## 4. Statistical Rigor

- [ ] Appropriate statistical tests chosen for the data distribution
- [ ] Multiple comparisons accounted for
- [ ] Confidence intervals reported alongside point estimates
- [ ] Sample sizes justify the conclusions drawn
- [ ] Effect sizes reported, not just p-values

## 5. Testing

- [ ] Unit tests cover core logic and edge cases
- [ ] Integration tests validate component interactions
- [ ] Data validation tests check schema and constraints
- [ ] Model tests verify predictions on known inputs (smoke tests)
- [ ] No test pollution — tests are independent and idempotent
- [ ] Test names describe the behavior being verified

## 6. Performance & Efficiency

- [ ] No unnecessary copies of large data structures
- [ ] Vectorized operations preferred over loops for numerical code
- [ ] Memory usage considered for large datasets (streaming, chunking)
- [ ] I/O operations are batched where possible
- [ ] Complexity is appropriate — no premature optimization, no obvious bottlenecks

## 7. Security & Privacy

- [ ] No secrets, API keys, or credentials in code
- [ ] PII is handled according to policy (masked, anonymized, or excluded)
- [ ] Input validation prevents injection or malformed data
- [ ] Dependencies are pinned and free of known vulnerabilities

## 8. Findings

### Blocking Issues

| # | File:Line | Issue | Suggestion |
|---|-----------|-------|------------|
| 1 | | | |

### Non-Blocking Suggestions

| # | File:Line | Suggestion | Priority |
|---|-----------|------------|----------|
| 1 | | | Low/Med |

### Positive Highlights

- [What was done well]

## 9. Overall Assessment

[Summary judgment: strengths, concerns, and conditions for approval]
