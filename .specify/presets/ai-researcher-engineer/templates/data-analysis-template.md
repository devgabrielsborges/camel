# Data Analysis: [ANALYSIS TITLE]

**Created**: [DATE]
**Analyst**: [NAME]
**Status**: In Progress | Complete | Peer Reviewed
**Related Spec**: [SPEC-ID or N/A]
**Notebook / Script**: [LINK]

---

## 1. Objective

State the question this analysis answers. Be precise enough that two analysts would produce comparable outputs.

> **Question**: [Your question here]

### 1.1 Context & Motivation

Why is this analysis needed? What decision will it inform?

### 1.2 Scope

- **Population**: [Who / what the data represents]
- **Time range**: [Date boundaries]
- **Granularity**: [e.g. per-user, per-session, per-day]

## 2. Data Sources

| Source | Description | Grain | Size | Access |
|--------|-------------|-------|------|--------|
| [Source 1] | [What it contains] | [Row-level entity] | [N rows × M cols] | [DB / API / file] |

### 2.1 Data Dictionary

| Column | Type | Description | Nulls | Notes |
|--------|------|-------------|-------|-------|
| [col_1] | [int/str/float/date] | [What it represents] | [% null] | [Caveats] |

### 2.2 Data Quality Assessment

- [ ] Missing values quantified and strategy defined (drop / impute / flag)
- [ ] Duplicates checked
- [ ] Outlier detection performed (method: [IQR / Z-score / domain rules])
- [ ] Distribution of key variables inspected
- [ ] Data types verified
- [ ] Joins validated (expected vs actual row counts)

## 3. Methodology

### 3.1 Approach

Describe the analytical approach and justify why it is appropriate.

### 3.2 Assumptions

| # | Assumption | Justification | Validation |
|---|-----------|---------------|------------|
| 1 | [e.g. Normality of residuals] | [Why reasonable] | [How checked] |
| 2 | [e.g. Independence of observations] | [Why reasonable] | [How checked] |

### 3.3 Statistical Methods

| Method | Purpose | Implementation |
|--------|---------|----------------|
| [e.g. Linear regression] | [Modeling relationship between X and Y] | [statsmodels / sklearn / R] |
| [e.g. Bootstrap CI] | [Uncertainty quantification] | [scipy / custom] |

### 3.4 Feature Engineering

| Feature | Derivation | Rationale |
|---------|-----------|-----------|
| [Feature 1] | [How computed] | [Why useful] |

## 4. Exploratory Data Analysis

### 4.1 Univariate Analysis

Summary statistics and distributions for key variables.

| Variable | Mean | Median | Std | Min | Max | Skewness |
|----------|------|--------|-----|-----|-----|----------|
| [var_1] | | | | | | |

### 4.2 Bivariate / Multivariate Analysis

Key relationships, correlations, and interactions.

| Relationship | Method | Finding |
|-------------|--------|---------|
| [X vs Y] | [Scatter / correlation / chi-sq] | [Result] |

### 4.3 Key Visualizations

[Reference or embed plots — distributions, scatter plots, heatmaps, time series]

## 5. Results

### 5.1 Primary Findings

| # | Finding | Evidence | Confidence |
|---|---------|----------|------------|
| 1 | [Statement] | [Metric, p-value, CI] | [High/Med/Low] |
| 2 | [Statement] | [Metric, p-value, CI] | [High/Med/Low] |

### 5.2 Model Performance (if applicable)

| Model | R² / Accuracy | RMSE / F1 | AIC / BIC | Cross-Val Score |
|-------|--------------|-----------|-----------|-----------------|
| [Model 1] | | | | |

### 5.3 Effect Sizes & Practical Significance

Beyond statistical significance — are the effects large enough to matter?

| Effect | Size | Benchmark | Practical Impact |
|--------|------|-----------|------------------|
| [Effect 1] | [Cohen's d / OR / etc.] | [Small/Med/Large] | [Business interpretation] |

## 6. Sensitivity Analysis

How robust are the results to changes in assumptions, parameters, or data?

| Variation | Impact on Conclusion | Concern Level |
|-----------|---------------------|---------------|
| [Remove outliers] | [Conclusion holds / changes] | [Low/Med/High] |
| [Different time window] | [Conclusion holds / changes] | [Low/Med/High] |
| [Alternative method] | [Conclusion holds / changes] | [Low/Med/High] |

## 7. Limitations

- [Limitation 1]: [Impact on conclusions and how to interpret results despite this]
- [Limitation 2]: [Impact on conclusions]
- **Confounders not controlled for**: [List known confounders]

## 8. Recommendations

| Priority | Recommendation | Supporting Evidence | Expected Impact |
|----------|---------------|-------------------|-----------------|
| P0 | [Action to take] | [Finding #] | [Quantified impact] |
| P1 | [Action to take] | [Finding #] | [Quantified impact] |

## 9. Reproducibility

- [ ] Analysis code committed (path: `[path]`)
- [ ] Data snapshot or query saved
- [ ] Random seeds documented
- [ ] Environment captured
- [ ] Results match when re-run

## 10. Follow-Up Questions

- [ ] [Question raised by this analysis]
- [ ] [Deeper investigation needed]
