# Experiment: [EXPERIMENT NAME]

**Created**: [DATE]
**Author**: [NAME]
**Status**: Proposed | Running | Complete | Abandoned
**Experiment ID**: EXP-[NUMBER]
**Related Spec**: [SPEC-ID or N/A]
**Tracking**: [MLflow run ID / W&B link / spreadsheet]

---

## 1. Hypothesis

State a falsifiable hypothesis. Follow the format: "If we [change X], then [metric Y] will [improve/decrease] by [amount], because [reasoning]."

> **H0** (null): [No effect statement]
> **H1** (alternative): [Expected effect statement]

## 2. Motivation

Why is this experiment worth running? Link to the research question or product need.

## 3. Experimental Design

### 3.1 Independent Variables

| Variable | Values / Levels | Rationale |
|----------|----------------|-----------|
| [e.g. learning rate] | [1e-3, 1e-4, 1e-5] | [Why these values] |

### 3.2 Dependent Variables (Metrics)

| Metric | Type | How Measured | Success Threshold |
|--------|------|-------------|-------------------|
| [Primary metric] | Primary | [Method] | [Threshold] |
| [Secondary metric] | Secondary | [Method] | [Threshold] |
| [Guardrail metric] | Guardrail | [Method] | [Must not degrade below X] |

### 3.3 Control & Baselines

| Baseline | Description | Expected Performance |
|----------|-------------|---------------------|
| [Baseline 1] | [Current production model / random / heuristic] | [Known metrics] |

### 3.4 Statistical Plan

- **Test type**: [t-test / Mann-Whitney / bootstrap / Bayesian]
- **Significance level (α)**: [e.g. 0.05]
- **Power (1-β)**: [e.g. 0.80]
- **Minimum detectable effect**: [e.g. 2% improvement in F1]
- **Sample size calculation**: [N required per group]
- **Multiple comparisons correction**: [Bonferroni / Holm / FDR / N/A]

## 4. Data

### 4.1 Training Data

- **Source**: [Dataset name, version, location]
- **Size**: [N samples]
- **Split strategy**: [train/val/test ratios, stratification]
- **Preprocessing**: [Steps applied]

### 4.2 Evaluation Data

- **Source**: [Holdout set, cross-validation folds]
- **Known biases**: [Distribution gaps, label noise]
- **Data version**: [Hash or tag for reproducibility]

## 5. Implementation

### 5.1 Environment

- **Hardware**: [GPU type, count, memory]
- **Framework**: [PyTorch / TensorFlow / JAX + versions]
- **Dependencies**: [requirements.txt / environment.yml hash]
- **Random seeds**: [Seed values used]

### 5.2 Training Configuration

```yaml
# Key hyperparameters (copy from config file or log here)
model: [architecture]
optimizer: [type]
learning_rate: [value]
batch_size: [value]
epochs: [value]
```

### 5.3 Reproducibility Checklist

- [ ] Code committed and tagged (git SHA: `[hash]`)
- [ ] Data version pinned
- [ ] Random seeds set for all sources of randomness
- [ ] Environment captured (pip freeze / conda export)
- [ ] Config file checked in alongside code
- [ ] Results logged to experiment tracker

## 6. Results

### 6.1 Quantitative Results

| Variant | Primary Metric | Secondary Metric | Guardrail | Training Time |
|---------|---------------|-----------------|-----------|---------------|
| Baseline | | | | |
| Variant A | | | | |
| Variant B | | | | |

### 6.2 Statistical Significance

| Comparison | Test | Statistic | p-value | Effect Size | CI (95%) | Significant? |
|------------|------|-----------|---------|-------------|----------|--------------|
| A vs Baseline | | | | | | |

### 6.3 Error Analysis

- **Failure modes**: [Where does the model fail?]
- **Confusion patterns**: [Systematic misclassifications]
- **Edge cases**: [Unexpected behaviors]

## 7. Discussion

- Does the evidence support or reject the hypothesis?
- What explains the observed results?
- What are the practical implications?
- What would you do differently?

## 8. Decision

> **Decision**: [Ship / Iterate / Abandon]
> **Rationale**: [Why this decision]
> **Next steps**: [Concrete follow-up actions]

## 9. Appendix

### A. Full Hyperparameter Sweep Results

[Link to experiment tracker or paste table]

### B. Learning Curves

[Link to plots or embed]

### C. Additional Visualizations

[Confusion matrices, attention maps, t-SNE, etc.]
