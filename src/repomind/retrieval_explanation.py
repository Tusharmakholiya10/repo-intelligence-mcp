from __future__ import annotations

from dataclasses import dataclass

from repomind.retrieval_strategy import RetrievalStrategy
from repomind.roles import ComponentRoleClassifier


@dataclass(frozen=True)
class RankingFactor:
    """One weighted factor contributing to a retrieval score."""

    name: str
    raw_score: float
    weight: float
    contribution: float


def build_ranking_explanation(
    *,
    similarity: float,
    lexical_score: float,
    implementation_score: float,
    symbol_score: float,
    contextual_role_score: float,
    strategy: RetrievalStrategy,
    query_text: str | None,
    component_role: str,
) -> dict:
    """
    Build a deterministic explanation for a semantic-search result.

    The explanation reports the weighted contribution of every ranking
    factor and identifies the strongest contributors.
    """

    factors = [
        RankingFactor(
            name="semantic_similarity",
            raw_score=similarity,
            weight=strategy.similarity_weight,
            contribution=(
                strategy.similarity_weight
                * similarity
            ),
        )
    ]

    if query_text:
        factors.extend(
            [
                RankingFactor(
                    name="lexical_relevance",
                    raw_score=lexical_score,
                    weight=strategy.lexical_weight,
                    contribution=(
                        strategy.lexical_weight
                        * lexical_score
                    ),
                ),
                RankingFactor(
                    name="implementation_relevance",
                    raw_score=implementation_score,
                    weight=strategy.implementation_weight,
                    contribution=(
                        strategy.implementation_weight
                        * implementation_score
                    ),
                ),
                RankingFactor(
                    name="symbol_relevance",
                    raw_score=symbol_score,
                    weight=strategy.symbol_weight,
                    contribution=(
                        strategy.symbol_weight
                        * symbol_score
                    ),
                ),
                RankingFactor(
                    name="contextual_role_relevance",
                    raw_score=contextual_role_score,
                    weight=strategy.contextual_weight,
                    contribution=(
                        strategy.contextual_weight
                        * contextual_role_score
                    ),
                ),
            ]
        )

    factors.sort(
        key=lambda factor: factor.contribution,
        reverse=True,
    )

    top_factors = factors[:3]

    reasons = []

    for factor in top_factors:
        if factor.contribution <= 0:
            continue

        reasons.append(
            {
                "factor": factor.name,
                "contribution": round(
                    factor.contribution,
                    4,
                ),
            }
        )

    architecture_match = False

    if query_text:
        query_roles = (
            ComponentRoleClassifier.classify_query(
                query_text
            )
        )

        architecture_match = (
            component_role in query_roles
        )

    return {
        "primary_factor": (
            top_factors[0].name
            if top_factors
            else None
        ),
        "factors": [
            {
                "name": factor.name,
                "raw_score": round(
                    factor.raw_score,
                    4,
                ),
                "weight": round(
                    factor.weight,
                    4,
                ),
                "contribution": round(
                    factor.contribution,
                    4,
                ),
            }
            for factor in factors
        ],
        "top_contributors": reasons,
        "architecture_role_match": architecture_match,
        "component_role": component_role,
    }