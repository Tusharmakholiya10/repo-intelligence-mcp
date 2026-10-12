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

    # Add child records so deletion verifies every cascade relationship.
    with indexer._connect() as connection:
        file_row = connection.execute(
            "SELECT id FROM files WHERE path = ?",
            ("example.py",),
        ).fetchone()
        assert file_row is not None
        file_id = file_row["id"]

        connection.execute(
            """
            INSERT INTO symbols
                (file_id, name, qualified_name, type, line, end_line)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_id, "hello", "hello", "function", 1, 1),
        )

        connection.execute(
            """
            INSERT INTO symbol_references
                (file_id, symbol_name, line, reference_type)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, "hello", 2, "call"),
        )

        # Create a second file to verify dependency cleanup from
        # both the source-file and target-file relationships.
        connection.execute(
            """
            INSERT INTO files
                (path, language, size, modified_time, semantic_indexed)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("other.py", "python", 10, "2026-01-01T00:00:00", 0),
        )

        other_row = connection.execute(
            "SELECT id FROM files WHERE path = ?",
            ("other.py",),
        ).fetchone()
        other_id = other_row["id"]

        connection.executemany(
            """
            INSERT INTO dependencies
                (source_file_id, target_file_id, dependency_type, line)
            VALUES (?, ?, ?, ?)
            """,
            [
                (file_id, other_id, "import", 1),
                (other_id, file_id, "import", 2),
            ],
        )

    # Existing deletion code, unchanged.
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

    with indexer._connect() as connection:
        remaining_symbols = connection.execute(
            "SELECT COUNT(*) FROM symbols WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0]

        remaining_references = connection.execute(
            "SELECT COUNT(*) FROM symbol_references WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0]

        remaining_dependencies = connection.execute(
            """
            SELECT COUNT(*) FROM dependencies
            WHERE source_file_id = ? OR target_file_id = ?
            """,
            (file_id, file_id),
        ).fetchone()[0]

        remaining_chunks = connection.execute(
            "SELECT COUNT(*) FROM semantic_chunks WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0]

    assert remaining_symbols == 0
    assert remaining_references == 0
    assert remaining_dependencies == 0
    assert remaining_chunks == 0


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



def test_file_bundle_update_rolls_back_on_failure(tmp_path):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    source_file = repository_root / "example.py"
    source_file.write_text(
        "def old_function():\n    return 1\n",
        encoding="utf-8",
    )

    indexer = CodeIndexer(repository_root)

    old_symbol = {
        "name": "old_function",
        "qualified_name": "old_function",
        "type": "function",
        "line": 1,
        "end_line": 2,
    }

    old_reference = {
        "symbol_name": "old_function",
        "line": 4,
        "reference_type": "call",
    }

    old_chunk = {
        "start_line": 1,
        "end_line": 2,
        "symbol_name": "old_function",
        "symbol_type": "function",
        "content": "def old_function(): return 1",
        "embedding": [0.1, 0.2],
    }

    indexer.index_file_bundle(
        relative_path="example.py",
        language="python",
        size=source_file.stat().st_size,
        modified_time="2026-01-01T00:00:00",
        symbols=[old_symbol],
        references=[old_reference],
        dependencies=[],
        semantic_chunks=[old_chunk],
    )

    # The next update deletes and replaces several records
    # within its transaction, but then fails on the empty embedding.
    new_symbol = {
        "name": "new_function",
        "qualified_name": "new_function",
        "type": "function",
        "line": 1,
        "end_line": 2,
    }

    bad_chunk = {
        "start_line": 1,
        "end_line": 2,
        "symbol_name": "new_function",
        "symbol_type": "function",
        "content": "def new_function(): return 2",
        "embedding": [],
    }

    with pytest.raises(
        ValueError,
        match="missing embedding",
    ):
        indexer.index_file_bundle(
            relative_path="example.py",
            language="python",
            size=source_file.stat().st_size,
            modified_time="2026-01-02T00:00:00",
            symbols=[new_symbol],
            references=[],
            dependencies=[],
            semantic_chunks=[bad_chunk],
        )

    # The failed update must not replace the old index.
    symbols = indexer.search_symbols(
        "old_function",
        max_results=10,
    )
    assert any(
        item["qualified_name"] == "old_function"
        for item in symbols
    )

    assert indexer.search_symbols(
        "new_function",
        max_results=10,
    ) == []

    assert indexer.find_usages(
        "old_function",
        max_results=10,
    )

    assert not indexer.find_usages(
        "new_function",
        max_results=10,
    )

    chunks = indexer.get_semantic_chunks("example.py")
    assert len(chunks) == 1
    assert chunks[0]["content"] == old_chunk["content"]

    assert (
        indexer.needs_semantic_reindex("example.py")
        is False
    )

import sqlite3


def test_database_write_failure_preserves_previous_file_bundle(
    tmp_path,
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()

    source_file = repository_root / "example.py"
    source_file.write_text(
        "def old_function():\n    return 1\n",
        encoding="utf-8",
    )

    indexer = CodeIndexer(repository_root)

    old_symbol = {
        "name": "old_function",
        "qualified_name": "old_function",
        "type": "function",
        "line": 1,
        "end_line": 2,
    }

    old_reference = {
        "symbol_name": "old_function",
        "line": 4,
        "reference_type": "call",
    }

    old_chunk = {
        "start_line": 1,
        "end_line": 2,
        "symbol_name": "old_function",
        "symbol_type": "function",
        "content": "def old_function(): return 1",
        "embedding": [0.1, 0.2],
    }

    old_size = source_file.stat().st_size
    old_modified_time = "2026-01-01T00:00:00"

    indexer.index_file_bundle(
        relative_path="example.py",
        language="python",
        size=old_size,
        modified_time=old_modified_time,
        symbols=[old_symbol],
        references=[old_reference],
        dependencies=[],
        semantic_chunks=[old_chunk],
    )

    # Force SQLite to reject the new semantic chunk.
    with indexer._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_new_semantic_chunk
            BEFORE INSERT ON semantic_chunks
            WHEN NEW.content LIKE '%new_function%'
            BEGIN
                SELECT RAISE(ABORT, 'simulated database failure');
            END;
            """
        )

    new_symbol = {
        "name": "new_function",
        "qualified_name": "new_function",
        "type": "function",
        "line": 1,
        "end_line": 2,
    }

    new_chunk = {
        "start_line": 1,
        "end_line": 2,
        "symbol_name": "new_function",
        "symbol_type": "function",
        "content": "def new_function(): return 2",
        "embedding": [0.3, 0.4],
    }

    with pytest.raises(
        sqlite3.IntegrityError,
        match="simulated database failure",
    ):
        indexer.index_file_bundle(
            relative_path="example.py",
            language="python",
            size=old_size + 100,
            modified_time="2026-01-02T00:00:00",
            symbols=[new_symbol],
            references=[],
            dependencies=[],
            semantic_chunks=[new_chunk],
        )

    # The previous symbols remain; the new symbol is absent.
    old_symbols = indexer.search_symbols(
        "old_function",
        max_results=10,
    )
    assert any(
        item["qualified_name"] == "old_function"
        for item in old_symbols
    )

    assert indexer.search_symbols(
        "new_function",
        max_results=10,
    ) == []

    # The old references remain; no new references were committed.
    assert indexer.find_usages(
        "old_function",
        max_results=10,
    )
    assert not indexer.find_usages(
        "new_function",
        max_results=10,
    )

    # The old semantic chunk remains intact.
    chunks = indexer.get_semantic_chunks("example.py")
    assert len(chunks) == 1
    assert chunks[0]["content"] == old_chunk["content"]

    # Failed persistence must not commit the new file metadata.
    with indexer._connect() as connection:
        file_row = connection.execute(
            """
            SELECT language, size, modified_time, semantic_indexed
            FROM files
            WHERE path = ?
            """,
            ("example.py",),
        ).fetchone()

    assert file_row is not None
    assert file_row["language"] == "python"
    assert file_row["size"] == old_size
    assert file_row["modified_time"] == old_modified_time
    assert file_row["semantic_indexed"] == 1

    assert indexer.needs_semantic_reindex("example.py") is False