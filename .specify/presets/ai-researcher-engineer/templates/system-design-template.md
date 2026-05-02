# System Design: [SYSTEM NAME]

**Created**: [DATE]
**Author**: [NAME]
**Status**: Draft | In Review | Approved | Implemented
**Related Spec**: [SPEC-ID or N/A]

---

## 1. Overview

### 1.1 Purpose

What does this system do and why does it exist? One paragraph.

### 1.2 Scope

- **In scope**: [What this design covers]
- **Out of scope**: [What is explicitly excluded]

### 1.3 Key Design Goals

| Priority | Goal | Rationale |
|----------|------|-----------|
| P0 | [e.g. Serve predictions at p99 < 100ms] | [Why] |
| P1 | [e.g. Support model hot-swapping without downtime] | [Why] |
| P2 | [e.g. Minimize infrastructure cost] | [Why] |

## 2. Architecture

### 2.1 High-Level Diagram

```
[System diagram: data sources → ingestion → processing → model → serving → consumers]
```

### 2.2 Component Inventory

| Component | Responsibility | Technology | Owner |
|-----------|---------------|------------|-------|
| Data Ingestion | [What it does] | [e.g. Kafka, Airflow] | [Team] |
| Feature Store | [What it does] | [e.g. Feast, Redis] | [Team] |
| Training Pipeline | [What it does] | [e.g. Kubeflow, Argo] | [Team] |
| Model Registry | [What it does] | [e.g. MLflow, Vertex AI] | [Team] |
| Serving Layer | [What it does] | [e.g. TorchServe, Triton] | [Team] |
| Monitoring | [What it does] | [e.g. Prometheus, Grafana] | [Team] |

### 2.3 Data Flow

Describe the end-to-end data flow from raw input to prediction output.

1. **Ingestion**: [How data enters the system]
2. **Transformation**: [Feature engineering, preprocessing]
3. **Storage**: [Where processed data lives]
4. **Training**: [How models consume data]
5. **Serving**: [How predictions are generated]
6. **Feedback**: [How outcomes flow back for retraining]

## 3. Data Architecture

### 3.1 Data Sources

| Source | Format | Volume | Frequency | SLA |
|--------|--------|--------|-----------|-----|
| [Source 1] | [JSON/Parquet/CSV] | [GB/day] | [Real-time/Batch] | [Latency] |

### 3.2 Storage Strategy

| Layer | Technology | Data | Retention | Access Pattern |
|-------|------------|------|-----------|----------------|
| Raw | [S3/GCS] | [Unprocessed data] | [Duration] | [Write-heavy] |
| Processed | [Delta Lake/BigQuery] | [Features] | [Duration] | [Read-heavy] |
| Serving | [Redis/DynamoDB] | [Online features] | [Duration] | [Low-latency reads] |

### 3.3 Data Quality

- **Validation framework**: [Great Expectations / Deequ / custom]
- **Schema enforcement**: [How schemas are versioned and enforced]
- **Data drift detection**: [Method and alert thresholds]

## 4. ML Pipeline

### 4.1 Training Pipeline

```
[raw data] → [validation] → [feature eng.] → [train] → [evaluate] → [register] → [deploy]
```

- **Orchestrator**: [Airflow / Argo / Kubeflow]
- **Compute**: [GPU type, autoscaling policy]
- **Frequency**: [On-demand / scheduled / triggered]
- **Artifact storage**: [Where models, metrics, configs are stored]

### 4.2 Feature Engineering

| Feature | Source | Transformation | Freshness | Online/Offline |
|---------|--------|---------------|-----------|----------------|
| [Feature 1] | [Source] | [Logic] | [Real-time/Daily] | [Both] |

### 4.3 Model Management

- **Versioning**: [How models are versioned]
- **Registry**: [Where models are stored with metadata]
- **Promotion workflow**: [Staging → Canary → Production]
- **Rollback procedure**: [How to revert to previous version]

## 5. Serving Architecture

### 5.1 Inference Pattern

- [ ] Real-time (synchronous API)
- [ ] Near-real-time (streaming)
- [ ] Batch (scheduled)
- [ ] Edge / on-device

### 5.2 API Design

```
POST /v1/predict
{
  "input": { ... },
  "model_version": "optional"
}

Response:
{
  "prediction": { ... },
  "confidence": 0.95,
  "model_version": "v2.3.1",
  "latency_ms": 42
}
```

### 5.3 Scaling Strategy

- **Autoscaling**: [Metric-based: CPU, GPU, queue depth, latency]
- **Load balancing**: [Strategy]
- **Caching**: [What is cached, TTL, invalidation]
- **Warm-up**: [Model pre-loading strategy]

## 6. Monitoring & Observability

### 6.1 System Metrics

| Metric | Alert Threshold | Dashboard |
|--------|----------------|-----------|
| Latency (p50/p95/p99) | [Thresholds] | [Link] |
| Error rate | [Threshold] | [Link] |
| Throughput (req/s) | [Threshold] | [Link] |
| GPU utilization | [Threshold] | [Link] |

### 6.2 ML Metrics

| Metric | Monitoring Method | Alert Condition |
|--------|------------------|-----------------|
| Prediction distribution | [Statistical test] | [Drift threshold] |
| Feature distribution | [KL divergence / PSI] | [Drift threshold] |
| Ground truth accuracy | [Delayed labels] | [Degradation threshold] |
| Data quality | [Validation checks] | [Failure count] |

### 6.3 Logging & Tracing

- **Prediction logging**: [What is logged per prediction for debugging and retraining]
- **Distributed tracing**: [OpenTelemetry / Jaeger / X-Ray]
- **Audit trail**: [Compliance requirements]

## 7. Reliability

### 7.1 Failure Modes

| Failure | Detection | Impact | Mitigation |
|---------|-----------|--------|------------|
| Model server crash | [Health check] | [Degraded predictions] | [Auto-restart + fallback model] |
| Feature store unavailable | [Probe] | [Stale features] | [Cached features + graceful degradation] |
| Data pipeline delay | [SLA monitor] | [Stale model] | [Alert + use last known good model] |

### 7.2 Graceful Degradation

- **Fallback strategy**: [Simpler model / heuristic / cached predictions]
- **Circuit breakers**: [When to stop calling downstream services]
- **Timeout policy**: [Per-component timeout budgets]

## 8. Security & Privacy

- **Data encryption**: [At rest / in transit]
- **Access control**: [Who can access what]
- **PII handling**: [Anonymization, tokenization]
- **Model security**: [Adversarial robustness considerations]
- **Compliance**: [GDPR / HIPAA / SOC2 requirements]

## 9. Cost Analysis

| Component | Monthly Cost (est.) | Scaling Factor |
|-----------|-------------------|----------------|
| Compute (training) | [$X] | [Per training run] |
| Compute (inference) | [$X] | [Per 1M requests] |
| Storage | [$X] | [Per TB] |
| Data transfer | [$X] | [Per GB] |
| **Total** | **[$X]** | |

## 10. Clean Code Principles Applied

- [ ] Single Responsibility: each component has one clear purpose
- [ ] Dependency Inversion: abstractions over concrete implementations
- [ ] Interface Segregation: narrow, focused contracts between services
- [ ] Configuration as code: all settings version-controlled
- [ ] Immutable artifacts: models and data snapshots are never mutated
- [ ] Idempotent operations: safe to retry any pipeline step

## 11. Migration & Rollout

- **Phase 1**: [Shadow mode — run new system alongside old, compare outputs]
- **Phase 2**: [Canary — route X% of traffic]
- **Phase 3**: [Full rollout]
- **Rollback plan**: [Exact steps to revert]
- **Success criteria**: [Metrics that must be met to proceed to next phase]

## 12. Open Questions

- [ ] [Question 1]
- [ ] [Question 2]
