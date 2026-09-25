from repomind.retrieval_diversification import (
    diversify_results,
)


def test_duplicate_symbols_are_removed():
    results = [
        {
            "path": "src/indexer.py",
            "symbol_name": "semantic_search",
            "score": 1.0,
        },
        {
            "path": "src/indexer.py",
            "symbol_name": "semantic_search",
            "score": 0.9,
        },
        {
            "path": "src/server.py",
            "symbol_name": "semantic_search",
            "score": 0.8,
        },
    ]

    diversified = diversify_results(
        results,
        max_results=3,
    )

    assert len(diversified) == 2

    assert diversified[0]["path"] == (
        "src/indexer.py"
    )

    assert diversified[1]["path"] == (
        "src/server.py"
    )


def test_highest_ranked_duplicate_is_kept():
    results = [
        {
            "path": "src/indexer.py",
            "symbol_name": "semantic_search",
            "score": 1.2,
        },
        {
            "path": "src/indexer.py",
            "symbol_name": "semantic_search",
            "score": 0.7,
        },
    ]

    diversified = diversify_results(
        results,
        max_results=5,
    )

    assert len(diversified) == 1
    assert diversified[0]["score"] == 1.2


def test_symbolless_chunks_remain_unique():
    results = [
        {
            "path": "docs/example.txt",
            "symbol_name": None,
            "score": 0.9,
        },
        {
            "path": "docs/example.txt",
            "symbol_name": None,
            "score": 0.8,
        },
    ]

    diversified = diversify_results(
        results,
        max_results=5,
    )

    assert len(diversified) == 2


def test_max_results_is_respected():
    results = [
        {
            "path": f"src/file{i}.py",
            "symbol_name": f"function_{i}",
            "score": 1.0 - i * 0.01,
        }
        for i in range(10)
    ]

    diversified = diversify_results(
        results,
        max_results=3,
    )

    assert len(diversified) == 3