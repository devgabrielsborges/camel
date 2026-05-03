from __future__ import annotations

import math
from collections.abc import Callable

import nltk
import tiktoken
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from scipy.stats import spearmanr

from camel.domain.value_objects import Chunk
from camel.domain.value_objects.pass_at_k_result import PassAtKResult
from camel.domain.value_objects.score import Score

nltk.download("punkt_tab", quiet=True)

_ENCODING = tiktoken.get_encoding("o200k_base")

_STEMMERS: dict[str, SnowballStemmer] = {
    "english": SnowballStemmer("english"),
    "portuguese": SnowballStemmer("portuguese"),
    "spanish": SnowballStemmer("spanish"),
}

_NEGATION_STEMS: dict[str, frozenset[str]] = {
    "english": frozenset({"not", "n't", "no", "never", "neither"}),
    "portuguese": frozenset({"nã", "no", "nunc", "nem"}),
    "spanish": frozenset({"no", "nunc", "ni", "jamas"}),
}

_CAPABILITY_STEMS: dict[str, frozenset[str]] = {
    "english": frozenset({"have", "can", "find", "know", "possess", "provid", "help", "assist"}),
    "portuguese": frozenset({"tenh", "possu", "consig", "encontr", "pod", "sab", "hav"}),
    "spanish": frozenset({"teng", "pued", "encontr", "sab", "cont", "hav"}),
}

_SELF_NEGATING_STEMS: frozenset[str] = frozenset({"unabl", "imposs", "impossív"})

_REFUSAL_WINDOW = 4


def _stem_tokens(tokens: list[str], stemmer: SnowballStemmer) -> list[str]:
    return [stemmer.stem(t.lower()) for t in tokens]


def _has_refusal(stems: list[str], lang: str) -> bool:
    negations = _NEGATION_STEMS[lang]
    capabilities = _CAPABILITY_STEMS[lang]

    for stem in stems:
        if stem in _SELF_NEGATING_STEMS:
            return True

    for i, stem in enumerate(stems):
        if stem in negations:
            window = stems[max(0, i - _REFUSAL_WINDOW) : i + _REFUSAL_WINDOW + 1]
            if any(s in capabilities for s in window):
                return True
    return False


def _tokenize(text: str) -> list[int]:
    return _ENCODING.encode(text)


def token_overlap_f1(response: str, reference: str) -> Score:
    """Unigram F1 between response tokens and reference tokens."""
    response_tokens = set(_tokenize(response))
    reference_tokens = set(_tokenize(reference))

    if not response_tokens or not reference_tokens:
        return Score(scorer_name="token_overlap_f1", value=0.0)

    overlap = response_tokens & reference_tokens
    precision = len(overlap) / len(response_tokens)
    recall = len(overlap) / len(reference_tokens)

    if precision + recall == 0:
        return Score(scorer_name="token_overlap_f1", value=0.0)

    f1 = 2 * (precision * recall) / (precision + recall)
    return Score(scorer_name="token_overlap_f1", value=round(f1, 4))


def class_exact_match(response: str, expected_class_id: str, *, class_name: str = "") -> Score:
    """Check if the agent's response contains the expected class ID or class name."""
    response_lower = response.lower()
    match = expected_class_id.lower() in response_lower
    if not match and class_name:
        normalized = class_name.replace("_", " ").lower()
        match = normalized in response_lower
    return Score(
        scorer_name="class_exact_match",
        value=match,
        metadata={"expected": expected_class_id, "class_name": class_name},
    )


def refusal_detection(response: str) -> Score:
    """Detect refusal via NLTK tokenization + stemming across EN/PT/ES."""
    tokens = word_tokenize(response)
    for lang, stemmer in _STEMMERS.items():
        stems = _stem_tokens(tokens, stemmer)
        if _has_refusal(stems, lang):
            return Score(
                scorer_name="refusal_detection",
                value=True,
                metadata={"detected_language": lang},
            )
    return Score(scorer_name="refusal_detection", value=False)


def pass_at_k(
    question_id: str,
    responses: list[str],
    reference: str,
    scorer_fn: Callable[[str, str], Score],
    *,
    threshold: float = 0.3,
) -> PassAtKResult:
    """Evaluate k responses and determine if at least one exceeds the threshold.

    Uses the provided scorer_fn to score each response against the reference,
    then checks if any score meets or exceeds the threshold.
    """
    scores = [scorer_fn(response, reference) for response in responses]
    numeric_scores = [float(s.value) for s in scores if s.value is not None]
    best = max(numeric_scores) if numeric_scores else 0.0
    passed = best >= threshold

    return PassAtKResult(
        question_id=question_id,
        k=len(responses),
        responses=responses,
        scores=scores,
        passed=passed,
        best_score=round(best, 4),
    )


# ---------------------------------------------------------------------------
# Deterministic xAI metrics
# ---------------------------------------------------------------------------

_HEDGING_SOLO_STEMS: dict[str, frozenset[str]] = {
    "english": frozenset(
        {
            "mayb",
            "perhap",
            "possibl",
            "probabl",
            "unclear",
            "uncertain",
            "unsur",
            "appar",
        }
    ),
    "portuguese": frozenset(
        {
            "talvez",
            "possivel",
            "provavel",
            "incert",
            "aparent",
        }
    ),
    "spanish": frozenset(
        {
            "quiz",
            "quizas",
            "posibl",
            "probabl",
            "inciert",
            "aparent",
        }
    ),
}

_HEDGING_VERB_STEMS: dict[str, frozenset[str]] = {
    "english": frozenset({"think", "believ", "seem", "appear", "guess", "suppos", "doubt"}),
    "portuguese": frozenset({"acho", "parec", "duv", "acredit", "suponh"}),
    "spanish": frozenset({"cre", "parec", "dud", "supong"}),
}

_HEDGING_SUBJECT_STEMS: dict[str, frozenset[str]] = {
    "english": frozenset({"i", "it"}),
    "portuguese": frozenset({"eu", "me"}),
    "spanish": frozenset({"yo", "me"}),
}

_HEDGING_WINDOW = 3


def self_consistency(responses: list[str]) -> Score:
    """Pairwise token-overlap F1 across k responses. Reports mean agreement."""
    if len(responses) < 2:
        return Score(scorer_name="self_consistency", value=None)

    pairs: list[float] = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            s = token_overlap_f1(responses[i], responses[j])
            pairs.append(float(s.value or 0.0))

    mean_agreement = sum(pairs) / len(pairs)
    variance = sum((p - mean_agreement) ** 2 for p in pairs) / len(pairs)
    return Score(
        scorer_name="self_consistency",
        value=round(mean_agreement, 4),
        metadata={"variance": round(variance, 4), "n_pairs": len(pairs)},
    )


def _find_hedging_matches(stems: list[str], lang: str) -> list[dict[str, str]]:
    solo = _HEDGING_SOLO_STEMS[lang]
    verbs = _HEDGING_VERB_STEMS[lang]
    subjects = _HEDGING_SUBJECT_STEMS[lang]
    matches: list[dict[str, str]] = []
    seen: set[str] = set()

    for stem in stems:
        if stem in solo and stem not in seen:
            matches.append({"language": lang, "stem": stem, "type": "solo"})
            seen.add(stem)

    for i, stem in enumerate(stems):
        if stem in verbs and stem not in seen:
            window = stems[max(0, i - _HEDGING_WINDOW) : i + _HEDGING_WINDOW + 1]
            if any(s in subjects for s in window):
                matches.append({"language": lang, "stem": stem, "type": "verb+subject"})
                seen.add(stem)

    return matches


def hedging_detection(response: str) -> Score:
    """Detect hedging and uncertainty language via NLTK stemming across EN/PT/ES."""
    tokens = word_tokenize(response)
    all_matches: list[dict[str, str]] = []
    for lang, stemmer in _STEMMERS.items():
        stems = _stem_tokens(tokens, stemmer)
        all_matches.extend(_find_hedging_matches(stems, lang))
    return Score(
        scorer_name="hedging_detection",
        value=len(all_matches) > 0,
        metadata={"hedging_count": len(all_matches), "matches": all_matches},
    )


def question_response_overlap(response: str, question: str) -> Score:
    """Unigram F1 between the question and the response (topic relevancy proxy)."""
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


def response_length_ratio(response: str, reference: str) -> Score:
    """Ratio of response token count to reference content token count."""
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


def _lcs_length(a: list[int], b: list[int]) -> int:
    """Length of the longest common subsequence via DP."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(response: str, reference: str) -> Score:
    """LCS-based F1 (ROUGE-L) over tiktoken token IDs."""
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


def chunk_attribution(response: str, chunks: list[Chunk]) -> Score:
    """Per-chunk token overlap with the response.

    Reports max attribution, Shannon entropy of the attribution distribution,
    and Spearman correlation between chunk relevance scores and attribution.
    """
    if not chunks:
        return Score(scorer_name="chunk_attribution", value=None)

    attributions: list[float] = []
    relevance_scores: list[float] = []
    for chunk in chunks:
        f1 = token_overlap_f1(response, chunk.content)
        attributions.append(float(f1.value or 0.0))
        relevance_scores.append(chunk.score)

    max_attr = max(attributions)

    total = sum(attributions) or 1.0
    probs = [a / total for a in attributions]
    entropy = -sum(p * math.log2(p + 1e-10) for p in probs)

    corr = 0.0
    if len(attributions) > 2:
        result = spearmanr(relevance_scores, attributions)
        corr_val = float(result.statistic)
        if not math.isnan(corr_val):
            corr = corr_val

    return Score(
        scorer_name="chunk_attribution",
        value=round(max_attr, 4),
        metadata={
            "entropy": round(entropy, 4),
            "score_correlation": round(corr, 4),
            "per_chunk": [round(a, 4) for a in attributions],
        },
    )
