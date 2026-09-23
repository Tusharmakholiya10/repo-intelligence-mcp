from repomind.roles import ComponentRoleClassifier


def test_chunker_is_classified_as_chunking():
    role = ComponentRoleClassifier.classify_component(
        path="src/repomind/chunker.py",
        symbol_name="CodeChunker.chunk_python_file",
    )

    assert role == "chunking"


def test_embeddings_is_classified_as_embedding():
    role = ComponentRoleClassifier.classify_component(
        path="src/repomind/embeddings.py",
        symbol_name="EmbeddingEngine.embed_documents",
    )

    assert role == "embedding"


def test_repository_security_symbol_is_classified_as_security():
    role = ComponentRoleClassifier.classify_component(
        path="src/repomind/repository.py",
        symbol_name="_resolve_safe_path",
    )

    assert role == "security"


def test_indexer_dependency_symbol_is_classified_as_dependency():
    role = ComponentRoleClassifier.classify_component(
        path="src/repomind/indexer.py",
        symbol_name="index_dependencies",
    )

    assert role == "dependency"


def test_indexer_reference_symbol_is_classified_as_reference_tracking():
    role = ComponentRoleClassifier.classify_component(
        path="src/repomind/indexer.py",
        symbol_name="find_usages",
    )

    assert role == "reference_tracking"


def test_server_repository_indexing_is_orchestration():
    role = ComponentRoleClassifier.classify_component(
        path="src/repomind/server.py",
        symbol_name="index_repository",
    )

    assert role == "orchestration"


def test_query_identifies_embedding_role():
    roles = ComponentRoleClassifier.classify_query(
        "Where are repository code embeddings generated?"
    )

    assert "embedding" in roles


def test_pipeline_query_identifies_orchestration():
    roles = ComponentRoleClassifier.classify_query(
        "How are semantic chunks created and indexed?"
    )

    assert "chunking" in roles
    assert "indexing" in roles
    assert "orchestration" in roles