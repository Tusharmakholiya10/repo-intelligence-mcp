from repomind.retrieval_strategy import (
    get_retrieval_strategy,
)


def test_security_strategy():
    strategy = get_retrieval_strategy(
        "security"
    )

    assert strategy.name == "security"
    assert strategy.contextual_weight > 1.0


def test_symbol_strategy_prioritizes_symbols():
    strategy = get_retrieval_strategy(
        "symbol"
    )

    assert (
        strategy.symbol_weight
        > strategy.similarity_weight
    )


def test_architecture_strategy_prioritizes_context():
    strategy = get_retrieval_strategy(
        "architecture"
    )

    assert (
        strategy.contextual_weight
        > strategy.similarity_weight
    )


def test_unknown_intent_uses_general_strategy():
    strategy = get_retrieval_strategy(
        "unknown"
    )

    assert strategy.name == "general"