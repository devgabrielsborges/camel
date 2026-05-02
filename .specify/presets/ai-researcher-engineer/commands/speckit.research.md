---
description: "Conduct a structured research synthesis on an AI/ML topic"
---

## User Input

```text
$ARGUMENTS
```

You are conducting a research synthesis on an AI, ML, statistics, or data science topic. Follow this workflow:

### Step 1: Frame the Research Question

Transform the user's input into a precise, answerable research question. Clarify:
- What exactly do we need to know?
- What are the scope boundaries?
- What type of answer is expected? (comparison, best practice, feasibility, state-of-the-art survey)

### Step 2: Read the Template

Read `templates/research-template.md` to see the structure your output must follow.

### Step 3: Conduct the Research

Search systematically across available sources:
- **Codebase**: Check for existing implementations, experiments, or documentation
- **Prior specs**: Look for related research or experiment docs in the project
- **External knowledge**: Apply your training knowledge of papers, methods, and best practices

For each relevant source, extract:
- Key contribution or finding
- Methodology used
- Quantitative results
- Limitations and caveats
- Reproducibility (code/data availability)

### Step 4: Synthesize and Compare

Don't just list papers — synthesize:
1. **Compare methods** across dimensions: accuracy, latency, data requirements, complexity, production-readiness
2. **Identify consensus** vs. open debates in the field
3. **Spot gaps** where evidence is insufficient
4. **Evaluate practical applicability** to our specific context

### Step 5: Write the Document

Fill every section of the research template. Pay special attention to:

- **Statistical Considerations**: What evaluation methodology is appropriate? What are common statistical pitfalls in this domain?
- **Key Findings**: Distill to actionable, evidence-backed statements
- **Recommendations**: Rank by priority, confidence, and effort. Every recommendation must trace back to evidence.
- **Knowledge Gaps**: Be explicit about what we don't know

### Principles to Follow

- **Evidence over opinion**: Every claim needs a citation or data point.
- **Practical focus**: Research should lead to engineering decisions, not just academic summaries.
- **Honest uncertainty**: Clearly distinguish high-confidence findings from speculative ones.
- **Reproducibility**: Prefer methods with available code, clear methodology, and validated results.
- **Statistical literacy**: Report effect sizes, confidence intervals, and practical significance — not just p-values.
