from __future__ import annotations

import nltk
import tiktoken
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

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


def class_exact_match(response: str, expected_class_id: str) -> Score:
    """Check if the agent's response contains the expected class ID."""
    match = expected_class_id.lower() in response.lower()
    return Score(
        scorer_name="class_exact_match",
        value=match,
        metadata={"expected": expected_class_id},
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
