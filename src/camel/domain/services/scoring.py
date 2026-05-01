from __future__ import annotations

import re

import tiktoken

from camel.domain.value_objects.score import Score

_ENCODING = tiktoken.get_encoding("o200k_base")

_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)i don'?t have"),
    re.compile(r"(?i)i cannot"),
    re.compile(r"(?i)i'?m (not able|unable)"),
    re.compile(r"(?i)não possuo"),
    re.compile(r"(?i)não (tenho|consigo|posso)"),
    re.compile(r"(?i)no tengo"),
    re.compile(r"(?i)no (puedo|cuento con)"),
    re.compile(r"(?i)não (encontrei|há)"),
    re.compile(r"(?i)no (encontré|hay)"),
]


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
    """Detect refusal patterns in the agent response."""
    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(response):
            return Score(
                scorer_name="refusal_detection",
                value=True,
                metadata={"matched_pattern": pattern.pattern},
            )
    return Score(scorer_name="refusal_detection", value=False)
