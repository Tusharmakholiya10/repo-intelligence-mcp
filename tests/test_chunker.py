from repomind.chunker import CodeChunker, CodeChunk


def test_chunk_python_file_uses_symbols():
    source = """\
class Repository:
    def __init__(self):
        pass

    def read_file(self, path):
        return path
"""

    symbols = [
        {
            "name": "Repository",
            "qualified_name": "Repository",
            "type": "class",
            "line": 1,
            "end_line": 7,
        },
        {
            "name": "__init__",
            "qualified_name": "Repository.__init__",
            "type": "method",
            "line": 2,
            "end_line": 3,
        },
        {
            "name": "read_file",
            "qualified_name": "Repository.read_file",
            "type": "method",
            "line": 5,
            "end_line": 6,
        },
    ]

    chunker = CodeChunker()

    chunks = chunker.chunk_python_file(
        "src/repomind/repository.py",
        source,
        symbols,
    )

    assert chunks

    assert any(
        chunk.symbol_name == "Repository"
        for chunk in chunks
    )

    assert any(
        chunk.symbol_name == "Repository.__init__"
        for chunk in chunks
    )

    assert any(
        chunk.symbol_name == "Repository.read_file"
        for chunk in chunks
    )


def test_empty_source_returns_no_chunks():
    chunker = CodeChunker()

    chunks = chunker.chunk_python_file(
        "example.py",
        "",
        [],
    )

    assert chunks == []


def test_file_without_symbols_uses_fallback_chunks():
    source = "\n".join(
        f"line {number}"
        for number in range(1, 11)
    )

    chunker = CodeChunker(
        max_lines=4,
    )

    chunks = chunker.chunk_python_file(
        "example.py",
        source,
        [],
    )

    assert len(chunks) == 3

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 4

    assert chunks[1].start_line == 5
    assert chunks[1].end_line == 8

    assert chunks[2].start_line == 9
    assert chunks[2].end_line == 10


def test_large_symbol_is_split():
    source = "\n".join(
        f"line {number}"
        for number in range(1, 11)
    )

    symbols = [
        {
            "name": "large_function",
            "qualified_name": "large_function",
            "type": "function",
            "line": 1,
            "end_line": 10,
        }
    ]

    chunker = CodeChunker(
        max_lines=3,
    )

    chunks = chunker.chunk_python_file(
        "example.py",
        source,
        symbols,
    )

    assert len(chunks) == 4

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3

    assert chunks[1].start_line == 4
    assert chunks[1].end_line == 6

    assert chunks[2].start_line == 7
    assert chunks[2].end_line == 9

    assert chunks[3].start_line == 10
    assert chunks[3].end_line == 10


def test_build_embedding_text_includes_metadata():
    chunk = CodeChunk(
        file_path="src/repomind/repository.py",
        start_line=1,
        end_line=10,
        content="def _resolve_safe_path():\n    pass",
        symbol_name="Repository._resolve_safe_path",
        symbol_type="method",
    )

    embedding_text = CodeChunker.build_embedding_text(
        chunk
    )

    assert (
        "File: src/repomind/repository.py"
        in embedding_text
    )

    assert (
        "Symbol: Repository._resolve_safe_path"
        in embedding_text
    )

    assert (
        "Type: method"
        in embedding_text
    )

    assert (
        "def _resolve_safe_path()"
        in embedding_text
    )