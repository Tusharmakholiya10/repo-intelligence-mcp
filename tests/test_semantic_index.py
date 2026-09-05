from pathlib import Path

from repomind.indexer import CodeIndexer


def test_semantic_chunks_can_be_stored_and_retrieved(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    database_path = (
        repository_root
        / ".repomind"
        / "index.db"
    )

    source_file = (
        repository_root
        / "example.py"
    )

    source_file.write_text(
        "def hello():\n"
        "    return 'hello'\n",
        encoding="utf-8",
    )

    indexer = CodeIndexer(
        repository_root,
        database_path,
    )

    indexer.index_file(
        relative_path="example.py",
        language="python",
        size=source_file.stat().st_size,
        modified_time="2026-01-01T00:00:00",
        symbols=[],
    )

    chunks = [
        {
            "start_line": 1,
            "end_line": 2,
            "symbol_name": "hello",
            "symbol_type": "function",
            "content": (
                "def hello():\n"
                "    return 'hello'"
            ),
            "embedding": [0.1, 0.2, 0.3],
        }
    ]

    indexer.index_semantic_chunks(
        "example.py",
        chunks,
    )

    results = indexer.get_semantic_chunks(
        "example.py"
    )

    assert len(results) == 1

    result = results[0]

    assert result["path"] == "example.py"
    assert result["start_line"] == 1
    assert result["end_line"] == 2
    assert result["symbol_name"] == "hello"
    assert result["symbol_type"] == "function"
    assert result["content"] == (
        "def hello():\n"
        "    return 'hello'"
    )

    assert len(result["embedding"]) == 3

    assert result["embedding"][0] == 0.1
    assert result["embedding"][1] == 0.2
    assert result["embedding"][2] == 0.3


def test_semantic_chunks_are_replaced(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    indexer = CodeIndexer(
        repository_root
    )

    source_file = (
        repository_root
        / "example.py"
    )

    source_file.write_text(
        "print('hello')",
        encoding="utf-8",
    )

    indexer.index_file(
        relative_path="example.py",
        language="python",
        size=source_file.stat().st_size,
        modified_time="2026-01-01T00:00:00",
        symbols=[],
    )

    first_chunks = [
        {
            "start_line": 1,
            "end_line": 1,
            "content": "old content",
            "embedding": [0.1, 0.2],
        }
    ]

    indexer.index_semantic_chunks(
        "example.py",
        first_chunks,
    )

    second_chunks = [
        {
            "start_line": 1,
            "end_line": 1,
            "content": "new content",
            "embedding": [0.3, 0.4],
        }
    ]

    indexer.index_semantic_chunks(
        "example.py",
        second_chunks,
    )

    results = indexer.get_semantic_chunks(
        "example.py"
    )

    assert len(results) == 1
    assert results[0]["content"] == "new content"
    assert results[0]["embedding"][0] == 0.3


def test_semantic_chunk_stats(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    indexer = CodeIndexer(
        repository_root
    )

    source_file = (
        repository_root
        / "example.py"
    )

    source_file.write_text(
        "print('hello')",
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
                "end_line": 1,
                "content": "print('hello')",
                "embedding": [0.1, 0.2],
            }
        ],
    )

    stats = indexer.get_stats()

    assert stats["semantic_chunks"] == 1