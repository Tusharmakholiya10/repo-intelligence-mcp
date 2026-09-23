from repomind.query_intent import (
    QueryIntentClassifier,
)


def test_security_query():
    result = QueryIntentClassifier.classify(
        "Where are sensitive environment files protected?"
    )

    assert result.name == "security"


def test_dependency_query():
    result = QueryIntentClassifier.classify(
        "How are local file dependencies stored?"
    )

    assert result.name == "dependency"


def test_reference_query():
    result = QueryIntentClassifier.classify(
        "How does RepoMind find usages of a symbol?"
    )

    assert result.name == "reference"


def test_symbol_query():
    result = QueryIntentClassifier.classify(
        "Where are classes and functions stored?"
    )

    assert result.name == "symbol"


def test_architecture_query():
    result = QueryIntentClassifier.classify(
        "How are semantic chunks created and indexed?"
    )

    assert result.name == "architecture"


def test_empty_query():
    result = QueryIntentClassifier.classify("")

    assert result.name == "general"
    assert result.confidence == 0.0