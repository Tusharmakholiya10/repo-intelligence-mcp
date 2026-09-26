from repomind.query_reformulation import (
    extract_query_terms,
    reformulate_query,
)


def test_extract_query_terms():
    terms = extract_query_terms(
        "Where are repository code embeddings generated?"
    )

    assert "code" in terms
    assert "embeddings" in terms
    assert "generated" in terms
    assert "where" not in terms
    assert "repository" not in terms


def test_security_reformulation():
    result = reformulate_query(
        "Where is path traversal prevented?"
    )

    assert result.intent == "security"
    assert "_resolve_safe_path" not in (
        result.reformulated_query
    )

    assert "security" in (
        result.reformulated_query
    )


def test_semantic_search_reformulation():
    result = reformulate_query(
        "Where does RepoMind perform semantic code search?"
    )

    assert result.intent == "semantic_search"

    assert "semantic" in (
        result.reformulated_query
    )

    assert "search" in (
        result.reformulated_query
    )

    assert "implementation" in (
        result.reformulated_query
    )


def test_dependency_reformulation():
    result = reformulate_query(
        "How are local file dependencies stored?"
    )

    assert result.intent == "dependency"

    assert "dependency" in (
        result.reformulated_query
    )

    assert "imports" in (
        result.reformulated_query
    )


def test_architecture_reformulation():
    result = reformulate_query(
        "How are semantic chunks created and indexed?"
    )

    assert result.intent == "architecture"

    assert "workflow" in (
        result.reformulated_query
    )

    assert "components" in (
        result.reformulated_query
    )


def test_reformulation_is_deterministic():
    query = (
        "Where are repository code embeddings generated?"
    )

    first = reformulate_query(query)
    second = reformulate_query(query)

    assert (
        first.reformulated_query
        == second.reformulated_query
    )