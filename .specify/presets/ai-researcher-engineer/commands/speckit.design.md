---
description: "Architect an ML system end-to-end: data, training, serving, and monitoring"
---

## User Input

```text
$ARGUMENTS
```

You are designing an ML system architecture. Follow this workflow:

### Step 1: Understand the System

From the user's input, identify:
- What is the core ML capability being built? (classification, ranking, generation, recommendation, etc.)
- What are the serving requirements? (real-time, batch, streaming, edge)
- What are the scale expectations? (requests/second, data volume, model size)
- What are the reliability requirements? (uptime SLA, latency budget)
- What constraints exist? (cost, team size, existing infrastructure, compliance)

### Step 2: Read the Template

Read `templates/system-design-template.md` to see the structure your output must follow.

### Step 3: Explore the Codebase

Before designing from scratch:
- Check for existing infrastructure, services, and patterns in the codebase
- Identify reusable components and established conventions
- Note current technology choices and constraints
- Look for prior system-design docs

### Step 4: Design the System

Work through each section of the template:

#### Architecture (section 2)

Start with a high-level diagram showing:
- Data flow from source to prediction to consumer
- Component boundaries and responsibilities
- Synchronous vs asynchronous paths
- External dependencies

Each component must have a single clear responsibility (Single Responsibility Principle applied to system design).

#### Data Architecture (section 3)

Design for the reality that data quality is the #1 determinant of ML system quality:
- Separate raw, processed, and serving storage layers
- Schema enforcement at boundaries
- Data validation and drift detection built in from day one
- Clear lineage from raw data to features to predictions

#### ML Pipeline (section 4)

Design for reproducibility and iteration speed:
- Training pipeline must be fully automated and idempotent
- Feature engineering shared between training and serving (no train/serve skew)
- Model registry with versioning, metadata, and promotion workflow
- Clear rollback procedure

#### Serving (section 5)

Design for the serving pattern that matches requirements:
- API design with versioning
- Autoscaling strategy based on the right signals
- Caching strategy with proper invalidation
- Pre-loading and warm-up for cold start mitigation

#### Monitoring (section 6)

Two layers of monitoring — system and ML:
- **System**: latency, error rate, throughput, resource utilization
- **ML**: prediction distribution drift, feature drift, accuracy on delayed labels
- Every metric needs an alert threshold and a response playbook

#### Reliability (section 7)

Design for failure — because ML systems fail in unique ways:
- Model degradation is gradual and silent
- Feature pipelines can deliver stale data without erroring
- Data distribution shift can invalidate a model without any system alert

Build graceful degradation: fallback models, cached predictions, circuit breakers.

### Step 5: Apply Clean Code to System Design

- **Separation of Concerns**: Don't mix data ingestion, feature engineering, training, and serving in the same service
- **Dependency Inversion**: Components depend on abstractions (interfaces), not concrete implementations
- **Interface Segregation**: Narrow APIs between services — each service exposes only what consumers need
- **Open/Closed**: New models, features, or data sources can be added without modifying existing infrastructure
- **Configuration as Code**: Every setting, threshold, and parameter is version-controlled

### Principles to Follow

- **Start simple, design for extension**: Don't over-engineer. Build the simplest system that works, but with clean extension points.
- **Data-centric > model-centric**: Better data usually beats a better model. Invest in data quality infrastructure.
- **Observability is a feature**: If you can't measure it, you can't improve it. Build monitoring from day one.
- **Cost-aware design**: Every architectural decision has a cost implication. Make trade-offs explicit.
- **Fail gracefully**: ML systems must degrade, not crash. Always have a fallback.
- **Reproducibility**: Every prediction must be traceable to the model version, data version, and config that produced it.
