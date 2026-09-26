from __future__ import annotations

import re
from dataclasses import dataclass

from repomind.query_intent import (
    QueryIntentClassifier,
)


@dataclass(frozen=True)
class ReformulatedQuery:
    """Structured representation of a refined retrieval query."""

    original_query: str
    reformulated_query: str
    intent: str
    extracted_terms: tuple[str, ...]
    focus: str


FOCUS_TERMS = {
    "security": (
        "protection",
        "validation",
        "security",
    ),
    "semantic_search": (
        "semantic",
        "search",
        "implementation",
    ),
    "dependency": (
        "dependency",
        "imports",
    ),
    "reference": (
        "references",
        "usages",
        "calls",
    ),
    "symbol": (
        "symbol",
        "definition",
        "implementation",
    ),
    "architecture": (
        "workflow",
        "components",
        "implementation",
    ),
    "general": (
        "implementation",
        "code",
    ),
}


STOP_WORDS = {
    "where",
    "what",
    "which",
    "who",
    "when",
    "why",
    "how",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "and",
    "or",
    "with",
    "this",
    "that",
    "repository",
    "repo",
}


def extract_query_terms(
    query: str,
) -> tuple[str, ...]:
    """
    Extract meaningful lexical terms from a repository question.
    """

    normalized = query.lower()

    tokens = re.findall(
        r"[a-zA-Z_][a-zA-Z0-9_]*",
        normalized,
    )

    terms = []

    for token in tokens:
        if token in STOP_WORDS:
            continue

        if len(token) < 3:
            continue

        if token not in terms:
            terms.append(token)

    return tuple(terms)


def _select_focus(
    intent: str,
    terms: tuple[str, ...],
) -> str:
    """
    Determine the retrieval focus from query intent.
    """

    if intent == "security":
        return "security implementation"

    if intent == "semantic_search":
        return "semantic search implementation"

    if intent == "dependency":
        return "dependency relationships"

    if intent == "reference":
        return "symbol usages and references"

    if intent == "symbol":
        return "symbol definition and implementation"

    if intent == "architecture":
        return "architecture workflow"

    return "repository implementation"


def reformulate_query(
    query: str,
) -> ReformulatedQuery:
    """
    Produce a deterministic retrieval-oriented query.
    """

    normalized = " ".join(
        query.strip().split()
    ).rstrip("?")

    intent = QueryIntentClassifier.classify(
        normalized
    )

    terms = extract_query_terms(
        normalized
    )

    focus = _select_focus(
        intent.name,
        terms,
    )

    focus_terms = FOCUS_TERMS.get(
        intent.name,
        FOCUS_TERMS["general"],
    )

    merged_terms = []

    for term in (
        *terms,
        *focus_terms,
    ):
        if term not in merged_terms:
            merged_terms.append(term)

    reformulated = " ".join(
        merged_terms
    )

    return ReformulatedQuery(
        original_query=normalized,
        reformulated_query=reformulated,
        intent=intent.name,
        extracted_terms=terms,
        focus=focus,
    )