from repomind.context_packer import (
    pack_context,
)


def test_context_contains_agent_relevant_sections():
    result = pack_context(
        path="src/repomind/embeddings.py",
        symbol_name="EmbeddingEngine.embed_documents",
        source=(
            "class EmbeddingEngine:\n"
            "    def embed_documents(self, texts):\n"
            "        return texts\n"
        ),
        symbols=[
            {
                "type": "class",
                "name": "EmbeddingEngine",
                "qualified_name": "EmbeddingEngine",
                "line": 1,
                "end_line": 3,
            },
            {
                "type": "method",
                "name": "embed_documents",
                "qualified_name": (
                    "EmbeddingEngine.embed_documents"
                ),
                "line": 2,
                "end_line": 3,
            },
        ],
        usages=[
            {
                "path": "src/repomind/server.py",
                "line": 397,
                "reference_type": "method_call",
                "symbol_name": "embed_documents",
            }
        ],
        dependencies=[
            {
                "path": "src/repomind/server.py",
                "dependency_type": "import",
                "line": 10,
            }
        ],
    )

    assert result.symbol_name == (
        "EmbeddingEngine.embed_documents"
    )

    assert "=== PRIMARY ===" in result.content
    assert "=== SOURCE ===" in result.content
    assert "=== RELATED SYMBOLS ===" in result.content
    assert "=== USAGES ===" in result.content
    assert "=== DEPENDENCIES ===" in result.content