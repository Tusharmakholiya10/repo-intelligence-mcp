from repomind.agentic_retrieval import (
    build_second_pass_query,
    evaluate_retrieval,
    extract_retrieval_confidence,
)


def test_extract_retrieval_confidence():
    output = (
        "Retrieval confidence: 0.32 (low)"
    )

    confidence = (
        extract_retrieval_confidence(
            output
        )
    )

    assert confidence == (
        0.32,
        "low",
    )


def test_missing_confidence_does_not_retry():
    decision = evaluate_retrieval(
        "Where is semantic search implemented?",
        "No semantic matches found.",
    )

    assert decision.should_retry is False
    assert decision.next_query is None


def test_high_confidence_does_not_retry():
    decision = evaluate_retrieval(
        "Where is semantic search implemented?",
        (
            "Retrieval confidence: "
            "0.86 (high)"
        ),
    )

    assert decision.should_retry is False
    assert decision.confidence_score == 0.86


def test_low_confidence_triggers_second_pass():
    decision = evaluate_retrieval(
        "Where is semantic search implemented?",
        (
            "Retrieval confidence: "
            "0.31 (low)"
        ),
    )

    assert decision.should_retry is True
    assert decision.next_query is not None
    assert (
        "implementation"
        in decision.next_query
    )


def test_second_pass_query_is_deterministic():
    query = (
        "Where are repository code "
        "embeddings generated?"
    )

    first = build_second_pass_query(
        query
    )

    second = build_second_pass_query(
        query
    )

    assert first == second
    assert "implementation" in first


def test_security_query_gets_security_focused_suffix():
    query = (
        "Where are sensitive environment "
        "files protected?"
    )

    refined = build_second_pass_query(
        query
    )

    assert "validation" in refined
    assert "protection" in refined