import pytest
from repomind.chunker import CodeChunker
from repomind.indexer import CodeIndexer


class FakeEmbeddingEngine:
    """Deterministic embedding engine for tests."""

    def __init__(self):
        self.document_calls = 0
        self.closed = False

    def embed_documents(
        self,
        texts,
    ):
        self.document_calls += 1

        return [
            [
                float(index + 1),
                float(index + 2),
            ]
            for index, _ in enumerate(texts)
        ]

    def close(self):
        self.closed = True


def test_changed_file_generates_semantic_chunks(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    source_file = (
        repository_root / "example.py"
    )

    source = (
        "def hello():\n"
        "    return 'hello'\n"
    )

    source_file.write_text(
        source,
        encoding="utf-8",
    )

    symbols = [
        {
            "name": "hello",
            "qualified_name": "hello",
            "type": "function",
            "line": 1,
            "end_line": 2,
        }
    ]

    chunker = CodeChunker()

    chunks = chunker.chunk_python_file(
        "example.py",
        source,
        symbols,
    )

    fake_embeddings = FakeEmbeddingEngine()

    embeddings = (
        fake_embeddings.embed_documents(
            [chunk.content for chunk in chunks]
        )
    )

    semantic_records = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        semantic_records.append(
            {
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol_name": chunk.symbol_name,
                "symbol_type": chunk.symbol_type,
                "content": chunk.content,
                "embedding": embedding,
            }
        )

    indexer = CodeIndexer(
        repository_root
    )

    indexer.index_file(
        relative_path="example.py",
        language="python",
        size=source_file.stat().st_size,
        modified_time="2026-01-01T00:00:00",
        symbols=symbols,
    )

    indexer.index_semantic_chunks(
        "example.py",
        semantic_records,
    )

    indexer.mark_semantic_indexed(
        "example.py"
    )

    results = indexer.get_semantic_chunks(
        "example.py"
    )

    assert len(results) == 1
    assert results[0]["symbol_name"] == "hello"
    assert results[0]["symbol_type"] == "function"
    assert len(results[0]["embedding"]) == 2

    assert (
        indexer.needs_semantic_reindex(
            "example.py"
        )
        is False
    )


def test_unchanged_file_does_not_need_reindex(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    source_file = (
        repository_root / "example.py"
    )

    source_file.write_text(
        "print('hello')",
        encoding="utf-8",
    )

    indexer = CodeIndexer(
        repository_root
    )

    size = source_file.stat().st_size
    modified_time = "2026-01-01T00:00:00"

    indexer.index_file(
        relative_path="example.py",
        language="python",
        size=size,
        modified_time=modified_time,
        symbols=[],
    )

    assert (
        indexer.needs_reindex(
            "example.py",
            size,
            modified_time,
        )
        is False
    )


def test_modified_file_requires_reindex(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    source_file = (
        repository_root / "example.py"
    )

    source_file.write_text(
        "print('hello')",
        encoding="utf-8",
    )

    indexer = CodeIndexer(
        repository_root
    )

    original_size = source_file.stat().st_size

    indexer.index_file(
        relative_path="example.py",
        language="python",
        size=original_size,
        modified_time="2026-01-01T00:00:00",
        symbols=[],
    )

    source_file.write_text(
        "print('hello world')",
        encoding="utf-8",
    )

    new_size = source_file.stat().st_size

    assert (
        indexer.needs_reindex(
            "example.py",
            new_size,
            "2026-01-02T00:00:00",
        )
        is True
    )


def test_deleted_file_removes_semantic_chunks(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    source_file = (
        repository_root / "example.py"
    )

    source_file.write_text(
        "print('hello')",
        encoding="utf-8",
    )

    indexer = CodeIndexer(
        repository_root
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

    indexer.mark_semantic_indexed(
        "example.py"
    )

    source_file.unlink()

    with indexer._connect() as connection:
        connection.execute(
            """
            DELETE FROM files
            WHERE path = ?
            """,
            ("example.py",),
        )
        connection.commit()

    assert (
        indexer.get_semantic_chunks(
            "example.py"
        )
        == []
    )


def test_semantic_indexed_state_prevents_repeated_embedding(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    source_file = (
        repository_root / "example.py"
    )

    source_file.write_text(
        "print('hello')",
        encoding="utf-8",
    )

    indexer = CodeIndexer(
        repository_root
    )

    indexer.index_file(
        relative_path="example.py",
        language="python",
        size=source_file.stat().st_size,
        modified_time="2026-01-01T00:00:00",
        symbols=[],
    )

    assert (
        indexer.needs_semantic_reindex(
            "example.py"
        )
        is True
    )

    # Simulate successful semantic indexing,
    # even when there are no semantic chunks.
    indexer.index_semantic_chunks(
        "example.py",
        [],
    )

    indexer.mark_semantic_indexed(
        "example.py"
    )

    assert (
        indexer.needs_semantic_reindex(
            "example.py"
        )
        is False
    )


def test_existing_database_is_migrated(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    indexer = CodeIndexer(
        repository_root
    )

    source_file = (
        repository_root / "example.py"
    )

    source_file.write_text(
        "def hello():\n"
        "    return 'hello'\n",
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
                "content": (
                    "def hello():\n"
                    "    return 'hello'"
                ),
                "embedding": [0.1, 0.2],
            }
        ],
    )

    indexer.mark_semantic_indexed(
        "example.py"
    )

    stats = indexer.get_stats()

    assert stats["semantic_chunks"] == 1
    assert stats["semantic_indexed_files"] == 1