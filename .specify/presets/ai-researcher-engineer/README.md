# AI Researcher & Engineer

A Spec Kit preset for teams working at the intersection of **AI Research**, **Statistics**, **AI Engineering**, and **Data Science** — grounded in clean code principles and system design best practices.

## What This Preset Provides

### Templates

Structured document scaffolds for every stage of AI/ML work:

| Template | Purpose |
|----------|---------|
| `spec-template` | Feature specification with research backing, metrics, data requirements, and rollout planning |
| `research-template` | Literature review and research synthesis — systematic methodology, comparative analysis, and actionable recommendations |
| `experiment-template` | ML experiment design with hypothesis formulation, statistical plan, reproducibility checklist, and decision framework |
| `system-design-template` | End-to-end ML system architecture — data pipelines, training, serving, monitoring, reliability, and cost analysis |
| `code-review-template` | Code review checklist merging clean code principles (SOLID, naming, error handling) with ML-specific concerns (data leakage, train/serve skew, reproducibility) |
| `data-analysis-template` | Statistical analysis and data science investigation — EDA, methodology, sensitivity analysis, and reproducibility |
| `plan-template` | Phased project plan with research gates, milestones, risks, resources, and definition of done |

### Commands

AI agent workflows that guide structured thinking:

| Command | What It Does |
|---------|-------------|
| `speckit.specify` | Creates a feature spec by analyzing the problem, gathering research context, and filling every section with engineering rigor |
| `speckit.research` | Conducts a systematic research synthesis — frames the question, searches literature, compares methods, and distills actionable findings |
| `speckit.experiment` | Designs a reproducible experiment — formulates a falsifiable hypothesis, plans statistical tests, and pre-registers expected outcomes |
| `speckit.review` | Reviews code against clean code, system design, and ML best practices — produces specific, actionable, severity-ranked feedback |
| `speckit.design` | Architects an ML system end-to-end — data, training, serving, monitoring, reliability, and cost — applying SOLID principles at system scale |

## Core Principles

This preset encodes four pillars into every template and command:

**1. Research Rigor** — Every decision traces back to evidence. Templates require citations, comparative analysis, and honest assessment of knowledge gaps.

**2. Statistical Discipline** — Experiments demand pre-registered hypotheses, proper significance testing, effect sizes alongside p-values, and sensitivity analysis. No p-hacking, no cherry-picking.

**3. Clean Code & System Design** — SOLID principles applied from function-level (naming, SRP, no side effects) to system-level (dependency inversion, interface segregation, separation of concerns). Every spec includes testability and maintainability requirements.

**4. Production Thinking** — Specs and designs address scale, failure modes, monitoring, rollback, and cost from day one. No prototype-to-production gap.

## Usage

### Install the preset

```bash
specify preset add --dev ./ai-researcher-engineer
```

### Use a command

```bash
# Create a feature spec
specify specify "Build a real-time fraud detection model using transaction embeddings"

# Research a topic
specify research "Compare transformer architectures for time-series forecasting"

# Design an experiment
specify experiment "Test whether LoRA fine-tuning matches full fine-tuning on our classification task"

# Review code
specify review "Review the new feature pipeline in src/pipelines/feature_engineering.py"

# Design a system
specify design "Design the serving infrastructure for our recommendation engine"
```

### Verify resolution

```bash
specify preset resolve spec-template
specify preset resolve research-template
```

### Remove

```bash
specify preset remove ai-researcher-engineer
```

## Directory Structure

```
ai-researcher-engineer/
├── preset.yml                          # Manifest: metadata, templates, commands
├── README.md
├── templates/
│   ├── spec-template.md                # AI/ML feature specification
│   ├── research-template.md            # Research synthesis
│   ├── experiment-template.md          # Experiment design & tracking
│   ├── system-design-template.md       # ML system architecture
│   ├── code-review-template.md         # Clean code + ML review checklist
│   ├── data-analysis-template.md       # Statistical analysis
│   └── plan-template.md               # Project planning
└── commands/
    ├── speckit.specify.md              # Feature spec workflow
    ├── speckit.research.md             # Research synthesis workflow
    ├── speckit.experiment.md           # Experiment design workflow
    ├── speckit.review.md               # Code review workflow
    └── speckit.design.md              # System design workflow
```

## License

MIT
