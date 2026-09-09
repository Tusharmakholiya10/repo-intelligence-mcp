from repomind.indexer import CodeIndexer


def _index_chunks(indexer, file_path, chunks):
    indexer.index_file(
        relative_path=file_path,
        language="python",
        size=100,
        modified_time="2026-01-01T00:00:00",
        symbols=[],
    )

    indexer.index_semantic_chunks(
        relative_path=file_path,
        chunks=chunks,
    )

    indexer.mark_semantic_indexed(
        file_path
    )


def test_lexical_relevance_prefers_path_traversal_protection():
    correct = CodeIndexer._lexical_relevance(
        query="Where is path traversal prevented?",
        path="src/repomind/repository.py",
        symbol_name="_resolve_safe_path",
        symbol_type="method",
        content=(
            "Resolve a repository-relative path safely. "
            "Prevents path traversal outside the repository."
        ),
    )

    related = CodeIndexer._lexical_relevance(
        query="Where is path traversal prevented?",
        path="src/repomind/indexer.py",
        symbol_name="_normalize_path",
        symbol_type="method",
        content=(
            "Normalize repository-relative paths. "
            "SQLite stores paths using forward slashes."
        ),
    )

    assert correct > related


def test_hybrid_search_can_outrank_higher_embedding_similarity(tmp_path):
    indexer = CodeIndexer(
        repository_root=tmp_path,
        database_path=tmp_path / "index.db",
    )

    _index_chunks(
        indexer,
        "src/repomind/indexer.py",
        [
            {
                "start_line": 50,
                "end_line": 65,
                "symbol_name": "_normalize_path",
                "symbol_type": "method",
                "content": (
                    "Normalize repository-relative paths. "
                    "SQLite stores paths using forward slashes."
                ),
                "embedding": [0.90, 0.435889894],
            }
        ],
    )

    _index_chunks(
        indexer,
        "src/repomind/repository.py",
        [
            {
                "start_line": 815,
                "end_line": 840,
                "symbol_name": "_resolve_safe_path",
                "symbol_type": "method",
                "content": (
                    "Resolve a repository-relative path safely. "
                    "Prevents path traversal outside the repository."
                ),
                "embedding": [0.80, 0.60],
            }
        ],
    )

    results = indexer.semantic_search(
        query_embedding=[1.0, 0.0],
        query_text="Where is path traversal prevented?",
        max_results=2,
    )

    assert len(results) == 2
    assert results[0]["path"] == (
        "src/repomind/repository.py"
    )
    assert results[0]["symbol_name"] == (
        "_resolve_safe_path"
    )

    assert results[0]["score"] > results[1]["score"]
    assert results[0]["lexical_score"] > (
        results[1]["lexical_score"]
    )


def test_semantic_search_without_query_text_keeps_embedding_ranking(
    tmp_path,
):
    indexer = CodeIndexer(
        repository_root=tmp_path,
        database_path=tmp_path / "index.db",
    )

    _index_chunks(
        indexer,
        "first.py",
        [
            {
                "start_line": 1,
                "end_line": 3,
                "symbol_name": "first",
                "symbol_type": "function",
                "content": "First result.",
                "embedding": [0.99, 0.14106736],
            }
        ],
    )

    _index_chunks(
        indexer,
        "second.py",
        [
            {
                "start_line": 1,
                "end_line": 3,
                "symbol_name": "second",
                "symbol_type": "function",
                "content": "Second result.",
                "embedding": [0.70, 0.71414284],
            }
        ],
    )

    results = indexer.semantic_search(
        query_embedding=[1.0, 0.0],
        max_results=2,
    )

    assert results[0]["path"] == "first.py"
    assert results[0]["score"] == results[0]["similarity"]
    assert results[0]["lexical_score"] == 0.0
