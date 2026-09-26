from __future__ import annotations

import re
from dataclasses import dataclass

from repomind.query_intent import QueryIntentClassifier


CONFIDENCE_THRESHOLD = 0.45


@dataclass(frozen=True)
class RetrievalEscalation:
    """Decision about whether semantic retrieval needs another pass."""

    should_retry: bool
    confidence_score: float | None
    confidence_level: str | None
    next_query: str | None
    reason: str


INTENT_SUFFIXES = {
    "security": (
        "implementation validation protection"
    ),
    "semantic_search": (
        "implementation code"
    ),
    "dependency": (
        "implementation dependencies imports"
    ),
    "reference": (
        "implementation usages calls"
    ),
    "symbol": (
        "symbol definition implementation"
    ),
    "architecture": (
        "implementation workflow components"
    ),
    "general": (
        "implementation code"
    ),
}


def extract_retrieval_confidence(
    tool_output: str,
) -> tuple[float, str] | None:
    """
    Extract retrieval confidence from semantic_search output.

    Expected format:

        Retrieval confidence: 0.41 (low)
    """

    match = re.search(
        r"Retrieval confidence:\s*"
        r"([0-9]*\.?[0-9]+)\s*"
        r"\((low|medium|high)\)",
        tool_output,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return (
        float(match.group(1)),
        match.group(2).lower(),
    )


def build_second_pass_query(
    query: str,
) -> str:
    """
    Build a deterministic implementation-focused
    query for a second semantic retrieval pass.
    """

    normalized = " ".join(
        query.strip().split()
    ).rstrip("?")

    intent = QueryIntentClassifier.classify(
        normalized
    )

    suffix = INTENT_SUFFIXES.get(
        intent.name,
        INTENT_SUFFIXES["general"],
    )

    return f"{normalized} {suffix}"


def evaluate_retrieval(
    query: str,
    tool_output: str,
) -> RetrievalEscalation:
    """
    Decide whether a semantic retrieval result
    deserves an automatic second pass.
    """

    confidence = extract_retrieval_confidence(
        tool_output
    )

    if confidence is None:
        return RetrievalEscalation(
            should_retry=False,
            confidence_score=None,
            confidence_level=None,
            next_query=None,
            reason=(
                "Retrieval confidence was not "
                "available."
            ),
        )

    score, level = confidence

    if score >= CONFIDENCE_THRESHOLD:
        return RetrievalEscalation(
            should_retry=False,
            confidence_score=score,
            confidence_level=level,
            next_query=None,
            reason=(
                "Retrieval confidence is above "
                "the escalation threshold."
            ),
        )

    next_query = build_second_pass_query(
        query
    )

    if next_query == query:
        return RetrievalEscalation(
            should_retry=False,
            confidence_score=score,
            confidence_level=level,
            next_query=None,
            reason=(
                "A distinct second-pass query "
                "could not be generated."
            ),
        )

    return RetrievalEscalation(
        should_retry=True,
        confidence_score=score,
        confidence_level=level,
        next_query=next_query,
        reason=(
            "Retrieval confidence is below "
            "the escalation threshold."
        ),
    )