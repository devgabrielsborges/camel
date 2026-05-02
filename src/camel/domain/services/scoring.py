from __future__ import annotations

from collections.abc import Callable

import nltk
import tiktoken
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

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
