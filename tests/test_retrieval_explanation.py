from repomind.retrieval_explanation import (
    build_ranking_explanation,
)
from repomind.retrieval_strategy import (
    get_retrieval_strategy,
)


def test_semantic_only_explanation():
    strategy = get_retrieval_strategy(
        "general"
    )

    explanation = build_ranking_explanation(
        similarity=0.9,
        lexical_score=0.0,
        implementation_score=0.0,
        symbol_score=0.0,
        contextual_role_score=0.0,
        strategy=strategy,
        query_text=None,
        component_role="unknown",
    )

    assert (
        explanation["primary_factor"]
        == "semantic_similarity"
    )

    assert len(
        explanation["factors"]
    ) == 1


def test_hybrid_explanation_contains_all_factors():
    strategy = get_retrieval_strategy(
        "semantic_search"
    )

    explanation = build_ranking_explanation(
        similarity=0.8,
        lexical_score=0.7,
        implementation_score=0.1,
        symbol_score=0.12,
        contextual_role_score=0.1,
        strategy=strategy,
        query_text=(
            "Where is semantic search implemented?"
        ),
        component_role="search",
    )

    factor_names = {
        factor["name"]
        for factor in explanation["factors"]
    }

    assert "semantic_similarity" in factor_names
    assert "lexical_relevance" in factor_names
    assert "implementation_relevance" in factor_names
    assert "symbol_relevance" in factor_names
    assert "contextual_role_relevance" in factor_names


def test_factors_are_sorted_by_contribution():
    strategy = get_retrieval_strategy(
        "general"
    )

    explanation = build_ranking_explanation(
        similarity=0.2,
        lexical_score=0.9,
        implementation_score=0.2,
        symbol_score=0.3,
        contextual_role_score=0.4,
        strategy=strategy,
        query_text="find semantic search",
        component_role="search",
    )

    contributions = [
        factor["contribution"]
        for factor in explanation["factors"]
    ]

    assert contributions == sorted(
        contributions,
        reverse=True,
    )


def test_architecture_role_match():
    strategy = get_retrieval_strategy(
        "architecture"
    )

    explanation = build_ranking_explanation(
        similarity=0.8,
        lexical_score=0.5,
        implementation_score=0.1,
        symbol_score=0.1,
        contextual_role_score=0.2,
        strategy=strategy,
        query_text=(
            "Where is semantic search implemented?"
        ),
        component_role="search",
    )

    assert (
        explanation["architecture_role_match"]
        is True
    )