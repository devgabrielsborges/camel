---
description: "Review code against clean code, system design, and ML best practices"
---

## User Input

```text
$ARGUMENTS
```

You are performing a code review on AI/ML code with a focus on clean code principles, system design quality, and data science best practices. Follow this workflow:

### Step 1: Understand the Change

Read the code being reviewed. Identify:
- What does this change do?
- What is the scope — is it a feature, a fix, a refactor, or an experiment?
- What components does it touch?

### Step 2: Read the Review Template

Read `templates/code-review-template.md` to see the checklist structure.

### Step 3: Evaluate Against Checklists

Work through each checklist section systematically:

#### Clean Code (sections 2.1–2.5)

Apply Robert Martin's principles with pragmatism:
- **Naming**: Do names reveal intent? Would a new team member understand them?
- **Functions**: Are they small, focused, and side-effect free?
- **Error handling**: Specific exceptions, actionable messages, fail-fast?
- **Comments**: Does the code explain itself? Do comments add *why*, not *what*?
- **Structure**: Single Responsibility, Dependency Inversion, no god objects?

Be specific. Don't just say "naming could be better" — say what name is unclear and suggest an alternative.

#### ML / Data Science (section 3)

These are the most consequential bugs in ML code:
- **Data leakage**: Is information from the future or the test set leaking into training?
- **Reproducibility**: Can this be re-run with the same results?
- **Train/serve skew**: Are preprocessing steps identical between training and inference?
- **Silent failures**: Are there NaN propagations, shape mismatches, or numerical issues that could go undetected?
- **Hardcoded magic numbers**: Are hyperparameters externalized?

#### Statistical Rigor (section 4)

- Are the right tests being used for the data distribution?
- Are multiple comparisons accounted for?
- Are conclusions supported by the evidence (effect sizes, CIs, not just p-values)?

#### Testing (section 5)

- Unit tests for core logic and edge cases
- Integration tests for component interactions
- Model smoke tests (known input → expected output range)
- Data validation tests

#### Performance (section 6)

- Vectorized operations over loops for numerical code
- Memory efficiency for large datasets
- No unnecessary copies of tensors or dataframes

### Step 4: Write the Review

Fill the review template:
1. **Summary**: One paragraph on what the change does
2. **Checklists**: Mark each item as passed or note the issue
3. **Findings**: Separate blocking issues from non-blocking suggestions. Be specific — include file, line, and a concrete suggestion
4. **Positive highlights**: Call out what was done well
5. **Verdict**: Approve, Request Changes, or Needs Discussion

### Principles to Follow

- **Be specific and constructive**: "This function does too much" is weak. "This function handles parsing, validation, and persistence — consider splitting into three" is actionable.
- **Distinguish severity**: Blocking issues (bugs, data leakage, security) vs. suggestions (style, naming, optimization).
- **Assume good intent**: The author made rational choices — understand why before suggesting changes.
- **Teach, don't just catch**: When flagging an issue, explain the principle behind it so the author learns.
- **Praise good work**: Reinforcing good patterns is as important as catching bad ones.
