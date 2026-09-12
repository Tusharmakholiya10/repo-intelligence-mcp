import pytest

from repomind.indexer import CodeIndexer


def create_index(tmp_path):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    indexer = CodeIndexer(
        repository_root
    )

    source_file = (
        repository_root / "example.py"
    )

    source_file.write_text(
        "def secure_path():\n"
        "    return 'safe'\n",
        encoding="utf-8",
    )

    indexer.index_file(
        relative_path="example.py",
        language="python",
        size=source_file.stat().st_size,
        modified_time="2026-01-01T00:00:00",
        symbols=[],
    )

    indexer.index_semantic_chunks(
        "example.py",
        [
            {
                "start_line": 1,
                "end_line": 2,
                "symbol_name": "secure_path",
                "symbol_type": "function",
                "content": (
                    "def secure_path():\n"
                    "    return 'safe'"
                ),
                "embedding": [
                    1.0,
                    0.0,
                    0.0,
                ],
            },
            {
                "start_line": 4,
                "end_line": 5,
                "symbol_name": "git_history",
                "symbol_type": "function",
                "content": (
                    "def git_history():\n"
                    "    return 'commits'"
                ),
                "embedding": [
                    0.0,
                    1.0,
                    0.0,
                ],
            },
        ],
    )

    indexer.mark_semantic_indexed(
        "example.py"
    )

    return indexer


def test_semantic_search_returns_most_similar(
    tmp_path,
):
    indexer = create_index(tmp_path)

    results = indexer.semantic_search(
        query_embedding=[
            1.0,
            0.0,
            0.0,
        ],
        max_results=1,
    )

    assert len(results) == 1

    assert (
        results[0]["symbol_name"]
        == "secure_path"
    )

    assert (
        results[0]["similarity"]
        == pytest.approx(1.0)
    )


def test_semantic_search_orders_results(
    tmp_path,
):
    indexer = create_index(tmp_path)

    results = indexer.semantic_search(
        query_embedding=[
            0.8,
            0.2,
            0.0,
        ],
        max_results=2,
    )

    assert len(results) == 2

    assert (
        results[0]["symbol_name"]
        == "secure_path"
    )

    assert (
        results[0]["similarity"]
        >
        results[1]["similarity"]
    )


def test_semantic_search_respects_min_similarity(
    tmp_path,
):
    indexer = create_index(tmp_path)

    results = indexer.semantic_search(
        query_embedding=[
            1.0,
            0.0,
            0.0,
        ],
        max_results=10,
        min_similarity=0.95,
    )

    assert len(results) == 1

    assert (
        results[0]["symbol_name"]
        == "secure_path"
    )


def test_semantic_search_validates_query(
    tmp_path,
):
    indexer = create_index(tmp_path)

    with pytest.raises(ValueError):
        indexer.semantic_search(
            query_embedding=[]
        )


def test_semantic_search_validates_max_results(
    tmp_path,
):
    indexer = create_index(tmp_path)

    with pytest.raises(ValueError):
        indexer.semantic_search(
            query_embedding=[1.0, 0.0, 0.0],
            max_results=0,
        )


def test_semantic_search_validates_threshold(
    tmp_path,
):
    indexer = create_index(tmp_path)

    with pytest.raises(ValueError):
        indexer.semantic_search(
            query_embedding=[1.0, 0.0, 0.0],
            min_similarity=1.5,
        )