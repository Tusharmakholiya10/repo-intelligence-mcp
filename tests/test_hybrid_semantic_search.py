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
def test_implementation_relevance_prefers_production_code():
    production_score = CodeIndexer._implementation_relevance(
        query="Where is path traversal prevented?",
        path="src/repomind/repository.py",
        symbol_name="_resolve_safe_path",
        symbol_type="method",
    )

    test_score = CodeIndexer._implementation_relevance(
        query="Where is path traversal prevented?",
        path="tests/test_repository.py",
        symbol_name="test_path_traversal",
        symbol_type="function",
    )

    assert production_score > test_score


def test_implementation_relevance_does_not_penalize_test_queries():
    score = CodeIndexer._implementation_relevance(
        query="How is path traversal tested?",
        path="tests/test_repository.py",
        symbol_name="test_path_traversal",
        symbol_type="function",
    )

    assert score >= 0.0

def test_symbol_relevance_prefers_semantic_search_implementation():
    score = CodeIndexer._symbol_relevance(
        query="Where does RepoMind perform semantic code search?",
        symbol_name="semantic_search",
        symbol_type="function",
        path="src/repomind/indexer.py",
        content="Search semantic chunks using a hybrid relevance score.",
    )

    unrelated = CodeIndexer._symbol_relevance(
        query="Where does RepoMind perform semantic code search?",
        symbol_name="get_dependencies",
        symbol_type="function",
        path="src/repomind/indexer.py",
        content="Return local repository files that a source file depends on.",
    )

    assert score > unrelated


def test_symbol_relevance_prefers_embedding_implementation():
    score = CodeIndexer._symbol_relevance(
        query="Where are repository code embeddings generated?",
        symbol_name="EmbeddingEngine",
        symbol_type="class",
        path="src/repomind/embeddings.py",
        content="Generate semantic embeddings using Gemini.",
    )

    unrelated = CodeIndexer._symbol_relevance(
        query="Where are repository code embeddings generated?",
        symbol_name="index_repository",
        symbol_type="function",
        path="src/repomind/server.py",
        content="Build or update the SQLite code index.",
    )

    assert score > unrelated

def test_contextual_role_prefers_embedding_owner():
    owner_score = CodeIndexer._contextual_role_relevance(
        query="Where are repository code embeddings generated?",
        path="src/repomind/embeddings.py",
        symbol_name="EmbeddingEngine",
        symbol_type="class",
        content=(
            "Generate semantic embeddings using Gemini."
        ),
    )

    coordinator_score = CodeIndexer._contextual_role_relevance(
        query="Where are repository code embeddings generated?",
        path="src/repomind/server.py",
        symbol_name="index_repository",
        symbol_type="function",
        content=(
            "Build or update the SQLite code index."
        ),
    )

    assert owner_score > coordinator_score


def test_contextual_role_prefers_pipeline_coordinator():
    coordinator_score = CodeIndexer._contextual_role_relevance(
        query="How are semantic chunks created and indexed?",
        path="src/repomind/server.py",
        symbol_name="index_repository",
        symbol_type="function",
        content=(
            "Create semantic chunks, generate embeddings, "
            "and persist the semantic index."
        ),
    )

    storage_score = CodeIndexer._contextual_role_relevance(
        query="How are semantic chunks created and indexed?",
        path="src/repomind/indexer.py",
        symbol_name="index_semantic_chunks",
        symbol_type="method",
        content=(
            "Replace semantic chunks for a repository file."
        ),
    )

    assert coordinator_score > storage_score

def test_contextual_role_matches_chunker_module():
    score = CodeIndexer._contextual_role_relevance(
        query="Where is Python source code divided into semantic chunks?",
        path="src/repomind/chunker.py",
        symbol_name="CodeChunker.chunk_python_file",
        symbol_type="method",
        content="Create semantic chunks from Python source code.",
    )

    unrelated = CodeIndexer._contextual_role_relevance(
        query="Where is Python source code divided into semantic chunks?",
        path="src/repomind/indexer.py",
        symbol_name="CodeIndexer.index_semantic_chunks",
        symbol_type="method",
        content="Replace semantic chunks for a repository file.",
    )

    assert score > unrelated


def test_contextual_role_prefers_repository_orchestrator():
    score = CodeIndexer._contextual_role_relevance(
        query="How are semantic chunks created and indexed?",
        path="src/repomind/server.py",
        symbol_name="index_repository",
        symbol_type="function",
        content=(
            "Create semantic chunks, generate embeddings, "
            "and persist the semantic index."
        ),
    )

    low_level = CodeIndexer._contextual_role_relevance(
        query="How are semantic chunks created and indexed?",
        path="src/repomind/indexer.py",
        symbol_name="CodeIndexer.index_semantic_chunks",
        symbol_type="method",
        content="Replace semantic chunks for a repository file.",
    )

    assert score > low_level


def test_contextual_role_prefers_symbol_index_functions():
    score = CodeIndexer._contextual_role_relevance(
        query="Where are classes and functions stored in the code index?",
        path="src/repomind/indexer.py",
        symbol_name="index_file",
        symbol_type="method",
        content="Insert or replace a file and its symbols.",
    )

    unrelated = CodeIndexer._contextual_role_relevance(
        query="Where are classes and functions stored in the code index?",
        path="src/repomind/indexer.py",
        symbol_name="index_semantic_chunks",
        symbol_type="method",
        content="Replace semantic chunks for a repository file.",
    )

    assert score > unrelated

def test_contextual_role_uses_architecture_owner():
    embedding_score = (
        CodeIndexer._contextual_role_relevance(
            query=(
                "Where are repository code "
                "embeddings generated?"
            ),
            path="src/repomind/embeddings.py",
            symbol_name="EmbeddingEngine",
            symbol_type="class",
            content="Generate embeddings.",
        )
    )

    indexer_score = (
        CodeIndexer._contextual_role_relevance(
            query=(
                "Where are repository code "
                "embeddings generated?"
            ),
            path="src/repomind/indexer.py",
            symbol_name="index_file",
            symbol_type="method",
            content="Store indexed files.",
        )
    )

    assert embedding_score > indexer_score


def test_contextual_role_prefers_orchestration_for_pipeline_query():
    orchestration_score = (
        CodeIndexer._contextual_role_relevance(
            query=(
                "How are semantic chunks "
                "created and indexed?"
            ),
            path="src/repomind/server.py",
            symbol_name="index_repository",
            symbol_type="function",
            content="Coordinate repository indexing.",
        )
    )

    chunking_score = (
        CodeIndexer._contextual_role_relevance(
            query=(
                "How are semantic chunks "
                "created and indexed?"
            ),
            path="src/repomind/chunker.py",
            symbol_name="CodeChunker.chunk_python_file",
            symbol_type="method",
            content="Create semantic chunks.",
        )
    )

    assert orchestration_score > chunking_score

def test_semantic_search_returns_component_role(
    tmp_path,
):
    indexer = CodeIndexer(
        repository_root=tmp_path,
        database_path=tmp_path / "index.db",
    )

    _index_chunks(
        indexer,
        "src/repomind/embeddings.py",
        [
            {
                "start_line": 1,
                "end_line": 5,
                "symbol_name": "EmbeddingEngine",
                "symbol_type": "class",
                "content": (
                    "Generate repository "
                    "code embeddings."
                ),
                "embedding": [
                    1.0,
                    0.0,
                ],
            }
        ],
    )

    results = indexer.semantic_search(
        query_embedding=[
            1.0,
            0.0,
        ],
        query_text=(
            "Where are repository "
            "code embeddings generated?"
        ),
        max_results=1,
    )

    assert len(results) == 1
    assert results[0]["component_role"] == (
        "embedding"
    )