---
description: "Design a reproducible ML experiment with hypothesis, metrics, and baselines"
---

## User Input

```text
$ARGUMENTS
```

You are designing a machine learning experiment. Follow this workflow:

### Step 1: Clarify the Goal

From the user's input, identify:
- What are we trying to learn or prove?
- What is the independent variable (what we're changing)?
- What is the dependent variable (what we're measuring)?
- What prior work or baselines exist?

### Step 2: Read the Template

Read `templates/experiment-template.md` to see the structure your output must follow.

### Step 3: Formulate the Hypothesis

Write a falsifiable hypothesis in the standard format:
- **H0 (null)**: There is no effect / no difference
- **H1 (alternative)**: If we [change X], then [metric Y] will [improve/decrease] by [amount], because [reasoning]

The hypothesis must be specific enough to be tested with a clear success/failure outcome.

### Step 4: Design the Experiment

Fill the experimental design with rigor:

1. **Independent Variables**: Enumerate all variables being manipulated and their levels. Justify the chosen ranges.

2. **Metrics**: Define primary (what determines success), secondary (additional signal), and guardrail (must not degrade) metrics. Each metric needs a measurement method and threshold.

3. **Baselines**: Every experiment needs at least one baseline. Use the current production model, a published benchmark, or a simple heuristic.

4. **Statistical Plan**: This is critical — specify:
   - The statistical test (t-test, bootstrap, Bayesian, etc.) and why it's appropriate
   - Significance level and power
   - Minimum detectable effect size
   - Sample size calculation
   - Multiple comparisons correction if testing multiple variants

5. **Data**: Pin exact dataset versions, document splits, and note known biases.

6. **Reproducibility**: Fill the checklist — git SHA, random seeds, environment, config files.

### Step 5: Pre-Register Expected Results

Before running, document:
- What result would support the hypothesis?
- What result would reject it?
- What result would be inconclusive?
- What decision follows from each outcome? (ship / iterate / abandon)

### Principles to Follow

- **Pre-registration mindset**: Design the experiment fully before running it. No p-hacking.
- **One variable at a time**: Isolate the effect you're measuring unless using factorial design.
- **Reproducibility is non-negotiable**: Another engineer must be able to re-run and get the same results.
- **Practical significance > statistical significance**: A statistically significant result that doesn't matter practically is not useful.
- **Track everything**: Log to MLflow, W&B, or equivalent. If it's not logged, it didn't happen.
