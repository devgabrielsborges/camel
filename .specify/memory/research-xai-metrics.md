# Research Synthesis: Deterministic xAI Metrics for LLM Capability Evaluation

**Created**: 2026-05-03
**Researcher**: AI Research Agent
**Status**: Complete
**Related Spec**: 001-weni-eval-pipeline

---

## 1. Research Question

> **RQ**: What new **deterministic** (no LLM judge, fully reproducible, zero API cost) explainability metrics can we add to the evaluation pipeline to strengthen the capability verdict and provide deeper insight into *why* a model succeeds or fails at Q&A?

### 1.1 Scope & Boundaries

- **In scope**: Metrics that are (a) purely algorithmic — same input always produces same output, (b) require no additional LLM calls, (c) operate on data already available in the pipeline (`DatasetRecord` fields + model response), and (d) add explanatory value beyond lexical F1.
- **Out of scope**: LLM-as-judge metrics (Faithfulness, Answer Relevancy, etc.), weight-access metrics (SHAP, IG, LRP), anything requiring additional inference calls.
- **Time horizon**: Established NLP metrics + recent deterministic approaches (2023–2026).

### 1.2 Current Deterministic Metrics

| Metric | What It Computes | Gap |
|--------|-----------------|-----|
| `token_overlap_f1` | Unigram F1 (tiktoken IDs) between response and `content` | Lexical only — misses paraphrases, ignores structure |
| `class_exact_match` | Response contains `chosen_class_id` or class name | Binary — no partial credit, no wrong-class detection |
| `refusal_detection` | Negation stems near capability stems (EN/PT/ES) | Heuristic — misses hedging, verbose non-answers, over-qualifications |
| `pass_at_k` | Best F1 across k=3 responses ≥ 0.3 | Only uses F1; ignores semantic agreement/disagreement across responses |
| `failure_mode` | Rule-based classification from F1 + refusal + category | Post-hoc label, not a scored metric; thresholds are hard-coded |
| `discrimination_delta` | |mean_pos(F1) − mean_neg(F1)| | Single dimension; a model can "discriminate" by being bad at both |

## 2. Methodology

- **Approach**: Systematic review of deterministic NLP evaluation metrics applicable to Q&A with provided context, filtering for those that add independent signal beyond unigram F1.
- **Selection criteria**: (1) Deterministic, (2) implementable with standard Python libraries (nltk, tiktoken, scikit-learn, sentence-transformers, spacy), (3) leverages data already in `DatasetRecord` + `Trace`, (4) provides explainable output (not just a number).
- **Total candidate metrics reviewed**: 18
- **Recommended for implementation**: 8

## 3. Proposed Deterministic Metrics

### 3.1 ROUGE-L (Longest Common Subsequence F1)

**What it measures**: Phrase-level and ordering overlap via longest common subsequence between response and content. Unlike unigram F1 which treats tokens as a bag, ROUGE-L rewards preserving sequential structure.

**Why it matters**: A model that copies the right phrases in the right order vs. one that scatters matching tokens randomly will score the same on `token_overlap_f1` but differently on ROUGE-L. This captures *extraction quality* — a key xAI signal for "how well did the model locate and reproduce relevant information."

**Discrimination value**: For `positivo` samples, high ROUGE-L + high F1 = precise extraction. High F1 + low ROUGE-L = scattered token matching (possible hallucination with coincidental overlap). For `negativo`, ROUGE-L should be low — if high, the model is copying content it should be refusing.

**Implementation**: `rouge-score` library or manual LCS on tiktoken IDs. Pure Python, no model needed.

```python
def rouge_l(response: str, reference: str) -> Score:
    """LCS-based F1 between response and reference token sequences."""
    resp_tokens = _tokenize(response)
    ref_tokens = _tokenize(reference)
    if not resp_tokens or not ref_tokens:
        return Score(scorer_name="rouge_l", value=0.0)
    lcs_len = _lcs_length(resp_tokens, ref_tokens)
    precision = lcs_len / len(resp_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return Score(scorer_name="rouge_l", value=0.0)
    f1 = 2 * precision * recall / (precision + recall)
    return Score(scorer_name="rouge_l", value=round(f1, 4))
```

---

### 3.2 Self-Consistency Score (from existing pass@k responses)

**What it measures**: Pairwise token overlap agreement across the k=3 responses already generated. High agreement = model is stable/confident. Low agreement = model is uncertain, possibly confabulating.

**Why it matters**: This is a **deterministic approximation of semantic entropy** (Farquhar et al., Nature 2024) — the gold-standard hallucination detection method — but using data already in the pipeline with zero extra cost. The existing `pass_at_k` only checks `max(F1) >= threshold`; it discards the variance signal.

**Discrimination value**: For `positivo`, high self-consistency + high F1 = reliably capable. Low self-consistency = the model is guessing, even if one response is good. For `negativo`, high self-consistency in refusal = model reliably recognizes unanswerable questions. Low consistency = unstable behavior.

**Implementation**: Pairwise `token_overlap_f1` across k responses → mean and standard deviation.

```python
def self_consistency(responses: list[str]) -> Score:
    """Pairwise token overlap F1 across k responses. Reports mean agreement."""
    if len(responses) < 2:
        return Score(scorer_name="self_consistency", value=None)
    pairs = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            score = token_overlap_f1(responses[i], responses[j])
            pairs.append(float(score.value))
    mean_agreement = sum(pairs) / len(pairs)
    variance = sum((p - mean_agreement) ** 2 for p in pairs) / len(pairs)
    return Score(
        scorer_name="self_consistency",
        value=round(mean_agreement, 4),
        metadata={"variance": round(variance, 4), "n_pairs": len(pairs)},
    )
```

---

### 3.3 Chunk Attribution Score

**What it measures**: Per-chunk token overlap between the response and each chunk in `chunks_big`. Produces a distribution showing *which* chunks the model drew from and how concentrated or dispersed its attention was.

**Why it matters**: This is a deterministic explainability metric that answers "which parts of the knowledge base did the model use?" — directly relevant to the "explainability differentiator" requirement. It also detects when a model ignores high-relevance chunks (scored by `chunks_big[].score`) in favor of low-relevance ones.

**Discrimination value**: For `positivo`, the model should attribute heavily to high-score chunks. For `negativo`, attribution should be low/dispersed (the model shouldn't be drawing from content when the answer isn't there).

**Key derived signals**:
- `chunk_attribution_max`: highest per-chunk F1 (was the model focused?)
- `chunk_attribution_entropy`: Shannon entropy of the attribution distribution (dispersed vs. concentrated)
- `chunk_score_correlation`: Spearman correlation between chunk relevance scores and attribution F1s (is the model using the *right* chunks?)

```python
def chunk_attribution(response: str, chunks: list[Chunk]) -> Score:
    """Per-chunk token overlap with the response. Reports max, entropy, and
    correlation with chunk relevance scores."""
    if not chunks:
        return Score(scorer_name="chunk_attribution", value=None)
    attributions = []
    relevance_scores = []
    for chunk in chunks:
        f1 = token_overlap_f1(response, chunk.content)
        attributions.append(float(f1.value))
        relevance_scores.append(chunk.score)
    max_attr = max(attributions)
    # Shannon entropy of normalized attribution distribution
    total = sum(attributions) or 1.0
    probs = [a / total for a in attributions]
    entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
    # Spearman correlation: does the model use the most relevant chunks?
    if len(attributions) > 2:
        corr, _ = spearmanr(relevance_scores, attributions)
    else:
        corr = 0.0
    return Score(
        scorer_name="chunk_attribution",
        value=round(max_attr, 4),
        metadata={
            "entropy": round(entropy, 4),
            "score_correlation": round(corr, 4) if not math.isnan(corr) else 0.0,
            "per_chunk": [round(a, 4) for a in attributions],
        },
    )
```

---

### 3.4 Hedging Detection (extended refusal)

**What it measures**: Extends `refusal_detection` with patterns for hedging, over-qualification, and uncertainty language that falls short of explicit refusal. Covers EN/PT/ES.

**Why it matters**: Current `refusal_detection` catches "I cannot help" but misses "I'm not entirely sure, but it might be possible that..." — a verbose non-answer that inflates F1 (because it often parrots content tokens) while providing low-quality output. This is the most common failure mode missed by the current heuristic.

**Discrimination value**: For `positivo`, hedging + high F1 = model has the answer but lacks confidence (potentially capable but undertrained). For `negativo`, hedging is actually *desirable* — it shows appropriate uncertainty. The hedging rate differential (positivo vs. negativo) is a new discrimination signal.

```python
_HEDGING_PATTERNS: dict[str, list[str]] = {
    "english": [
        "not sure", "not certain", "might be", "could be", "possibly",
        "i think", "it seems", "it appears", "may not be accurate",
        "i don't have enough", "based on my understanding",
        "i cannot confirm", "it's unclear", "hard to say",
    ],
    "portuguese": [
        "não tenho certeza", "talvez", "pode ser", "possivelmente",
        "eu acho", "parece que", "não é claro", "difícil dizer",
        "não posso confirmar", "pelo que entendi",
    ],
    "spanish": [
        "no estoy seguro", "tal vez", "podría ser", "posiblemente",
        "creo que", "parece que", "no está claro", "difícil decir",
        "no puedo confirmar", "según entiendo",
    ],
}

def hedging_detection(response: str) -> Score:
    """Detect hedging and uncertainty language in the response."""
    response_lower = response.lower()
    detected = []
    for lang, patterns in _HEDGING_PATTERNS.items():
        for pattern in patterns:
            if pattern in response_lower:
                detected.append({"language": lang, "pattern": pattern})
    return Score(
        scorer_name="hedging_detection",
        value=len(detected) > 0,
        metadata={"hedging_count": len(detected), "matches": detected},
    )
```

---

### 3.5 Response-Question Overlap (Relevancy Proxy)

**What it measures**: Token overlap between the *question* and the *response* (not the content). A purely lexical proxy for "did the model address the topic being asked about?"

**Why it matters**: A model that produces a grammatically correct response drawn from content but completely ignores the question topic will score well on `token_overlap_f1` (response vs. content) but poorly on response-question overlap. This catches the "correct extraction, wrong question" failure mode.

**Discrimination value**: Combined with `token_overlap_f1`, this creates a 2D capability map:
- High content-F1 + high question-overlap = on-topic extraction (desired for `positivo`)
- High content-F1 + low question-overlap = off-topic extraction (failure)
- Low content-F1 + high question-overlap = on-topic but fabricated (hallucination)
- Low content-F1 + low question-overlap = completely irrelevant (off-topic)

```python
def question_response_overlap(response: str, question: str) -> Score:
    """Unigram F1 between the question and the response."""
    resp_tokens = set(_tokenize(response))
    q_tokens = set(_tokenize(question))
    if not resp_tokens or not q_tokens:
        return Score(scorer_name="question_response_overlap", value=0.0)
    overlap = resp_tokens & q_tokens
    precision = len(overlap) / len(resp_tokens)
    recall = len(overlap) / len(q_tokens)
    if precision + recall == 0:
        return Score(scorer_name="question_response_overlap", value=0.0)
    f1 = 2 * precision * recall / (precision + recall)
    return Score(scorer_name="question_response_overlap", value=round(f1, 4))
```

---

### 3.6 Response Length Ratio

**What it measures**: Ratio of response token count to content token count. Captures verbosity/conciseness relative to the available knowledge base.

**Why it matters**: Capable models for `positivo` should produce responses proportional to the relevant content — not 10x longer (filler/hallucination) or 10x shorter (incomplete). For `negativo`, capable models should produce short responses (refusals), so the ratio should be very low. This metric is trivial to compute and adds an orthogonal dimension.

**Discrimination value**: The length_ratio distribution for `positivo` vs. `negativo` should differ significantly for a capable model. If they're similar, the model isn't adjusting behavior based on content availability.

```python
def response_length_ratio(response: str, reference: str) -> Score:
    """Ratio of response length to reference content length (in tokens)."""
    resp_len = len(_tokenize(response))
    ref_len = len(_tokenize(reference))
    if ref_len == 0:
        return Score(scorer_name="response_length_ratio", value=None)
    ratio = resp_len / ref_len
    return Score(
        scorer_name="response_length_ratio",
        value=round(ratio, 4),
        metadata={"response_tokens": resp_len, "reference_tokens": ref_len},
    )
```

---

### 3.7 NLI Contradiction Score (lightweight model, deterministic)

**What it measures**: Natural language inference between response and content using a lightweight cross-encoder model (e.g., `cross-encoder/nli-deberta-v3-small`). Produces three class probabilities: entailment, neutral, contradiction. The contradiction probability is a hallucination signal.

**Why it matters**: Unlike `token_overlap_f1` which only measures token presence, NLI captures *semantic contradiction* — the model says X but the content says not-X. This is the deterministic equivalent of LLM-judge faithfulness: same concept, cheaper execution, fully reproducible.

**Discrimination value**: For `positivo`, contradiction should be near zero. For `negativo`, some contradiction may be acceptable (the model is explaining why it can't answer), but high entailment with content is a hallucination signal.

**Implementation**: Uses a sentence-transformers cross-encoder. Deterministic given fixed model + no temperature. Runs locally, ~50ms per sample on CPU.

```python
# Initialized once at module level
_NLI_MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-small")
_LABELS = ["contradiction", "entailment", "neutral"]

def nli_contradiction(response: str, reference: str) -> Score:
    """NLI contradiction probability between response and reference content."""
    if not response.strip() or not reference.strip():
        return Score(scorer_name="nli_contradiction", value=None)
    # Cross-encoder handles truncation internally
    scores = _NLI_MODEL.predict([(reference, response)])
    probs = softmax(scores[0])
    label_probs = dict(zip(_LABELS, probs))
    return Score(
        scorer_name="nli_contradiction",
        value=round(float(label_probs["contradiction"]), 4),
        metadata={
            "entailment": round(float(label_probs["entailment"]), 4),
            "neutral": round(float(label_probs["neutral"]), 4),
        },
    )
```

**Note**: This metric requires a small local model (~100MB). It's deterministic (no sampling) but requires a one-time download. If the "no model" constraint is strict, this becomes P2.

---

### 3.8 Instruction Compliance Score

**What it measures**: Lexical matching of explicit behavioral instructions from `DatasetRecord.instructions` against the response. Checks whether the response follows verifiable instructions (e.g., "Always mention X", "Answer only about Y", "Explain step by step").

**Why it matters**: The dataset includes per-agent behavioral `instructions` that are currently unused by any scorer. These are domain-specific constraints that a capable model must follow. Compliance is partially verifiable via keyword/pattern matching without a judge.

**Discrimination value**: Instruction compliance should be high across both `positivo` and `negativo` — a capable model follows behavioral rules regardless of whether the answer is available. Low compliance signals a fundamentally incapable model.

```python
def instruction_compliance(response: str, instructions: list[str]) -> Score:
    """Check verifiable instruction compliance via keyword presence."""
    if not instructions:
        return Score(scorer_name="instruction_compliance", value=None)
    response_lower = response.lower()
    verifiable = 0
    compliant = 0
    details = []
    for instr in instructions:
        # Extract key nouns/phrases from instruction
        keywords = _extract_instruction_keywords(instr)
        if keywords:
            verifiable += 1
            matched = any(kw.lower() in response_lower for kw in keywords)
            if matched:
                compliant += 1
            details.append({"instruction": instr, "compliant": matched})
    if verifiable == 0:
        return Score(scorer_name="instruction_compliance", value=None)
    rate = compliant / verifiable
    return Score(
        scorer_name="instruction_compliance",
        value=round(rate, 4),
        metadata={"verifiable": verifiable, "compliant": compliant, "details": details},
    )
```

---

## 4. Comparative Analysis

### 4.1 Full Metric Comparison

| Metric | Measures | Independence from F1 | Discrimination Signal | Compute Cost | Dependencies |
|--------|---------|---------------------|----------------------|-------------|-------------|
| **rouge_l** | Phrase-level extraction quality | Medium (correlated but not redundant with unigram F1) | Shows extraction precision vs. scattered matching | ~0ms | tiktoken (already used) |
| **self_consistency** | Response stability across k=3 | High (orthogonal dimension) | Uncertainty/confabulation proxy | ~0ms (data exists) | None new |
| **chunk_attribution** | Which knowledge chunks were used | High (uses chunks_big, not content) | Explains partial answers; validates retrieval use | ~1ms | scipy.stats |
| **hedging_detection** | Uncertainty language | High (refusal misses hedging) | Calibration; hedging-rate differential | ~0ms | None new |
| **question_response_overlap** | Topic relevancy | High (question vs. content are different) | Off-topic detection | ~0ms | tiktoken (already used) |
| **response_length_ratio** | Verbosity/conciseness | High (fully orthogonal) | Length-behavior discrimination | ~0ms | tiktoken (already used) |
| **nli_contradiction** | Semantic contradiction | Very High (semantic, not lexical) | Direct hallucination detection | ~50ms/sample | sentence-transformers |
| **instruction_compliance** | Behavioral rule following | Very High (uses instructions field) | Instruction adherence across categories | ~0ms | nltk (already used) |

### 4.2 Expected Correlation Matrix (pre-implementation hypothesis)

|                        | F1   | ROUGE-L | Self-cons | Chunk-attr | Hedging | Q-R overlap | Length | NLI  | Instr |
|------------------------|------|---------|-----------|------------|---------|-------------|--------|------|-------|
| **token_overlap_f1**   | 1.0  | ~0.8    | ~0.3      | ~0.6       | ~-0.2   | ~0.3        | ~0.2   | ~-0.4| ~0.2  |
| **rouge_l**            |      | 1.0     | ~0.3      | ~0.5       | ~-0.2   | ~0.3        | ~0.1   | ~-0.3| ~0.2  |
| **self_consistency**   |      |         | 1.0       | ~0.2       | ~-0.3   | ~0.1        | ~0.1   | ~-0.3| ~0.1  |
| **chunk_attribution**  |      |         |           | 1.0        | ~-0.1   | ~0.2        | ~0.3   | ~-0.3| ~0.1  |
| **hedging_detection**  |      |         |           |            | 1.0     | ~0.1        | ~0.2   | ~0.2 | ~-0.1 |
| **q_r_overlap**        |      |         |           |            |         | 1.0         | ~0.1   | ~-0.1| ~0.2  |
| **length_ratio**       |      |         |           |            |         |             | 1.0    | ~0.1 | ~0.0  |
| **nli_contradiction**  |      |         |           |            |         |             |        | 1.0  | ~-0.1 |
| **instruction_compl**  |      |         |           |            |         |             |        |      | 1.0   |

ROUGE-L is the only metric likely to be highly correlated with F1 (ρ~0.8). All others add substantially independent signal.

## 5. Statistical Considerations

- **Integration path**: All new metrics return `Score` objects and can be added to `_STATISTICAL_METRICS` in `score_collector.py`. The existing `run_all_tests` + BH correction infrastructure handles them automatically.
- **Verdict integration**: New critical metric keys for the statistical verdict:
  - `("self_consistency", "global")` — low consistency → not capable
  - `("nli_contradiction", "positivo")` — high contradiction → hallucination
  - `("hedging_detection", "positivo")` — high hedging on positive → under-confident
  - `("chunk_attribution", "positivo")` — low max attribution → not using context
- **Correlation validation**: After implementation, compute Spearman ρ between all metric pairs. Drop any with ρ > 0.85 against an existing metric (keep the more interpretable one).
- **Threshold derivation**: Run `threshold_derivation` CLI on reference models with new metrics to establish empirical thresholds before incorporating into verdicts.
- **Effect sizes**: Require Cohen's d ≥ 0.5 between `positivo` and `negativo` distributions for a new metric to be included as a critical verdict key.

## 6. Key Findings

1. **Self-consistency is the highest-value deterministic addition.** The data for k=3 responses already exists in the pipeline. Computing pairwise agreement costs nothing and provides a deterministic approximation of semantic entropy — the gold-standard hallucination detection method (Farquhar et al., Nature 2024). This transforms `pass_at_k` from a binary "did any pass?" into a rich stability signal.

2. **Chunk attribution provides the most direct explainability.** The dataset's `chunks_big` field with per-chunk relevance scores is currently unused by any scorer. Chunk attribution answers "which parts of the knowledge base did the model use?" — exactly the kind of explainability that the benchmark values as a differentiator. The Spearman correlation between chunk relevance and response overlap is a novel discrimination metric.

3. **NLI contradiction is the only deterministic semantic metric.** All current metrics are lexical. A lightweight NLI cross-encoder (~100MB, ~50ms) adds the semantic dimension that `token_overlap_f1` fundamentally cannot provide: detecting when the model says something that *contradicts* the content, even if it uses different words.

4. **Hedging detection fills the gap between "refusal" and "answer".** Current failure mode classification jumps from "refusal" to "partial answer" with nothing in between. Real model outputs often contain hedging language ("I think maybe...", "it's possible that...") which is neither a refusal nor a confident answer. This creates a new failure mode category.

5. **Response-question overlap is trivially cheap and surprisingly informative.** It catches a failure mode invisible to content-F1: the model extracts content faithfully but from the wrong topic. Combined with content-F1, it creates a 2D capability map that separates four distinct behaviors.

6. **ROUGE-L adds limited independent signal over F1 but improves failure mode resolution.** The ρ~0.8 correlation means it won't revolutionize the verdict, but it distinguishes precise extraction (high ROUGE-L, high F1) from scattered token matching (low ROUGE-L, high F1) — useful for qualitative analysis in the report.

## 7. Recommendations

| Priority | Metric | Confidence | Effort | Expected Impact |
|----------|--------|------------|--------|-----------------|
| P0 | **self_consistency** — pairwise F1 across k=3 responses | High | S | Strongest new discrimination signal; zero extra cost; approximates semantic entropy |
| P0 | **chunk_attribution** — per-chunk F1 + entropy + score correlation | High | S | Direct explainability; uses untapped `chunks_big` data; novel signal |
| P1 | **nli_contradiction** — cross-encoder NLI for semantic contradiction | High | M | Only semantic metric; catches contradictory hallucinations invisible to lexical F1 |
| P1 | **hedging_detection** — extended multilingual hedging patterns | High | S | Fills refusal-answer gap; new failure mode category; near-zero cost |
| P1 | **question_response_overlap** — question-response F1 | High | S | Off-topic detection; trivial implementation; creates 2D capability map with content-F1 |
| P2 | **response_length_ratio** — response/content token ratio | Medium | S | Orthogonal signal; trivial to implement; improves verbose-answer detection |
| P2 | **rouge_l** — LCS-based F1 | Medium | S | Refines extraction quality analysis; moderate correlation with existing F1 |
| P2 | **instruction_compliance** — keyword-based instruction verification | Medium | M | Uses untapped `instructions` field; partial coverage (not all instructions are lexically verifiable) |

### Implementation Order

```
Phase 1 (zero new dependencies):
  self_consistency → hedging_detection → question_response_overlap → response_length_ratio

Phase 2 (scipy only — already likely in deps):
  chunk_attribution → rouge_l

Phase 3 (sentence-transformers dependency):
  nli_contradiction

Phase 4 (design work needed for keyword extraction):
  instruction_compliance
```

### Verdict Integration

```python
_STATISTICAL_METRICS = frozenset({
    "token_overlap_f1",
    "refusal_detection",
    "class_exact_match",
    "groundedness",
    "pass_at_k",
    # Phase 1
    "self_consistency",
    "hedging_detection",
    "question_response_overlap",
    "response_length_ratio",
    # Phase 2
    "chunk_attribution",
    "rouge_l",
    # Phase 3
    "nli_contradiction",
})
```

## 8. Knowledge Gaps

- [ ] **Self-consistency threshold**: What mean pairwise F1 across k=3 indicates "stable" vs. "unstable"? Needs empirical calibration on the WeniEval dataset.
- [ ] **NLI model language coverage**: `cross-encoder/nli-deberta-v3-small` is primarily English-trained. Need to evaluate or replace with a multilingual variant (`cross-encoder/nli-distilroberta-base` or `joeddav/xlm-roberta-large-xnli`) for PT/ES samples.
- [ ] **Chunk attribution on flat content**: Some records may have a single large chunk rather than multiple scored chunks. Need a fallback behavior (skip metric or subdivide content).
- [ ] **Hedging pattern completeness**: The initial patterns are manually curated. Need validation against actual model outputs to refine and avoid false positives.
- [ ] **Instruction keyword extraction**: Not all instructions contain verifiable keywords ("Be helpful" is not lexically checkable). Need to define which instruction types are amenable to deterministic verification.
- [ ] **Correlation with existing metrics**: Hypothesized correlations in section 4.2 must be validated empirically. If any new metric correlates ρ > 0.85 with F1, it provides explainability but not discriminative value.

## 9. References

1. Farquhar, S. et al. (2024). Detecting hallucinations in LLMs using semantic entropy. *Nature*, 630, 625–630. https://doi.org/10.1038/s41586-024-07421-0
2. Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop*.
3. Parcalabescu, L. & Frank, A. (2024). On Measuring Faithfulness or Self-consistency of NL Explanations. *ACL 2024*. https://aclanthology.org/2024.acl-long.329/
4. Williams, A. et al. (2018). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. *NAACL*. (NLI foundations)
5. Applied XAI for LLMs: Comparative Study (2026). https://arxiv.org/abs/2604.15371
6. BELL: Benchmarking Explainability of LLMs (2025). https://arxiv.org/abs/2504.18572
7. Reimers, N. & Gurevych, I. (2019). Sentence-BERT for semantic similarity. https://www.sbert.net/
8. Conneau, A. et al. (2018). XNLI: Evaluating cross-lingual sentence representations. *EMNLP*.
