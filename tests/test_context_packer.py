from repomind.context_packer import (
    pack_context,
)


def test_packs_primary_symbol_and_source():
    source = "\n".join(
        [
            "import os",
            "",
            "class Demo:",
            "    def run(self):",
            "        value = 42",
            "        return value",
        ]
    )

    symbols = [
        {
            "type": "class",
            "name": "Demo",
            "qualified_name": "Demo",
            "line": 3,
            "end_line": 6,
        },
        {
            "type": "method",
            "name": "run",
            "qualified_name": "Demo.run",
            "line": 4,
            "end_line": 6,
        },
    ]

    context = pack_context(
        path="src/demo.py",
        symbol_name="Demo.run",
        source=source,
        symbols=symbols,
        usages=[],
        dependencies=[],
    )

    assert "=== PRIMARY ===" in context.content
    assert "=== SOURCE ===" in context.content
    assert "Demo.run" in context.content
    assert "value = 42" in context.content


def test_includes_related_symbols():
    symbols = [
        {
            "type": "function",
            "name": "first",
            "qualified_name": "first",
            "line": 1,
            "end_line": 3,
        },
        {
            "type": "function",
            "name": "second",
            "qualified_name": "second",
            "line": 5,
            "end_line": 7,
        },
    ]

    context = pack_context(
        path="src/example.py",
        symbol_name="first",
        source="def first():\n    return 1\n\ndef second():\n    return 2\n",
        symbols=symbols,
        usages=[],
        dependencies=[],
    )

    assert "=== RELATED SYMBOLS ===" in context.content
    assert "second" in context.content


def test_includes_usages_and_dependencies():
    context = pack_context(
        path="src/server.py",
        symbol_name="EmbeddingEngine",
        source="class EmbeddingEngine:\n    pass\n",
        symbols=[
            {
                "type": "class",
                "name": "EmbeddingEngine",
                "qualified_name": "EmbeddingEngine",
                "line": 1,
                "end_line": 2,
            }
        ],
        usages=[
            {
                "path": "src/server.py",
                "line": 20,
                "reference_type": "call",
                "symbol_name": "EmbeddingEngine",
            }
        ],
        dependencies=[
            {
                "path": "src/embeddings.py",
                "dependency_type": "import",
                "line": 10,
            }
        ],
    )

    assert "=== USAGES ===" in context.content
    assert "src/server.py:20" in context.content
    assert "=== DEPENDENCIES ===" in context.content
    assert "src/embeddings.py" in context.content


def test_context_respects_max_chars():
    source = "\n".join(
        f"line {index}"
        for index in range(1, 500)
    )

    context = pack_context(
        path="large.py",
        symbol_name=None,
        source=source,
        symbols=[],
        usages=[],
        dependencies=[],
        max_chars=1000,
    )

    assert (
        context.total_characters
        <= 1000
    )

    assert context.truncated is True