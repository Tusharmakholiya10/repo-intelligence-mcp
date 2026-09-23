from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalStrategy:
    """Weights and behavior used for one retrieval intent."""

    name: str
    similarity_weight: float
    lexical_weight: float
    implementation_weight: float
    symbol_weight: float
    contextual_weight: float


STRATEGIES = {
    "security": RetrievalStrategy(
        name="security",
        similarity_weight=0.58,
        lexical_weight=0.30,
        implementation_weight=1.05,
        symbol_weight=1.05,
        contextual_weight=1.10,
    ),
    "semantic_search": RetrievalStrategy(
        name="semantic_search",
        similarity_weight=0.68,
        lexical_weight=0.27,
        implementation_weight=0.95,
        symbol_weight=0.95,
        contextual_weight=0.95,
    ),
    "dependency": RetrievalStrategy(
        name="dependency",
        similarity_weight=0.55,
        lexical_weight=0.30,
        implementation_weight=1.00,
        symbol_weight=1.10,
        contextual_weight=1.00,
    ),
    "reference": RetrievalStrategy(
        name="reference",
        similarity_weight=0.55,
        lexical_weight=0.30,
        implementation_weight=1.00,
        symbol_weight=1.10,
        contextual_weight=1.00,
    ),
    "symbol": RetrievalStrategy(
        name="symbol",
        similarity_weight=0.50,
        lexical_weight=0.30,
        implementation_weight=1.00,
        symbol_weight=1.15,
        contextual_weight=1.00,
    ),
    "architecture": RetrievalStrategy(
        name="architecture",
        similarity_weight=0.58,
        lexical_weight=0.27,
        implementation_weight=0.95,
        symbol_weight=0.95,
        contextual_weight=1.15,
    ),
    "general": RetrievalStrategy(
        name="general",
        similarity_weight=0.62,
        lexical_weight=0.30,
        implementation_weight=1.00,
        symbol_weight=1.00,
        contextual_weight=1.00,
    ),
}


def get_retrieval_strategy(
    intent: str,
) -> RetrievalStrategy:
    """Return the strategy for an intent."""

    return STRATEGIES.get(
        intent,
        STRATEGIES["general"],
    )