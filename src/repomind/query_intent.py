from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryIntent:
    name: str
    confidence: float


class QueryIntentClassifier:
    """Classify repository questions into retrieval intents."""

    INTENTS = {
        "security",
        "semantic_search",
        "dependency",
        "reference",
        "symbol",
        "architecture",
        "general",
    }

    SECURITY_TERMS = {
        "security",
        "secure",
        "protected",
        "protect",
        "sensitive",
        "secret",
        "env",
        "environment",
        "traversal",
        "access denied",
    }

    DEPENDENCY_TERMS = {
        "dependency",
        "dependencies",
        "depends on",
        "imported by",
        "imports",
    }

    REFERENCE_TERMS = {
        "usage",
        "usages",
        "reference",
        "references",
        "where used",
        "called by",
        "calls",
    }

    SYMBOL_TERMS = {
        "class",
        "classes",
        "function",
        "functions",
        "method",
        "methods",
        "symbol",
        "symbols",
    }

    ARCHITECTURE_TERMS = {
        "architecture",
        "component",
        "module",
        "pipeline",
        "workflow",
        "how does",
        "how are",
    }

    SEARCH_TERMS = {
        "where",
        "find",
        "search",
        "locate",
        "semantic",
        "implementation",
    }

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(
            query.lower().strip().split()
        )

    @classmethod
    def classify(
        cls,
        query: str,
    ) -> QueryIntent:
        """Return the highest-confidence query intent."""

        normalized = cls._normalize(query)

        if not normalized:
            return QueryIntent(
                name="general",
                confidence=0.0,
            )

        def count_matches(
            terms: set[str],
        ) -> int:
            return sum(
                term in normalized
                for term in terms
            )

        security_hits = count_matches(
            cls.SECURITY_TERMS
        )

        dependency_hits = count_matches(
            cls.DEPENDENCY_TERMS
        )

        reference_hits = count_matches(
            cls.REFERENCE_TERMS
        )

        symbol_hits = count_matches(
            cls.SYMBOL_TERMS
        )

        architecture_hits = count_matches(
            cls.ARCHITECTURE_TERMS
        )

        search_hits = count_matches(
            cls.SEARCH_TERMS
        )

        scores = {
            "security": security_hits,
            "dependency": dependency_hits,
            "reference": reference_hits,
            "symbol": symbol_hits,
            "architecture": architecture_hits,
            "semantic_search": search_hits,
        }

        intent, hits = max(
            scores.items(),
            key=lambda item: item[1],
        )

        if hits == 0:
            return QueryIntent(
                name="general",
                confidence=0.25,
            )

        confidence = min(
            1.0,
            0.5 + 0.15 * hits,
        )

        return QueryIntent(
            name=intent,
            confidence=confidence,
        )