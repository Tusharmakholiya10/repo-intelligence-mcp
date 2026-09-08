from examples.agent import (
    is_quota_exhausted_error,
    is_semantic_query,
    is_transient_gemini_error,
)


def test_quota_exhaustion_is_not_retryable():
    error = Exception(
        "429 RESOURCE_EXHAUSTED: quota exceeded "
        "for generate_content_free_tier_requests"
    )

    assert is_quota_exhausted_error(error)
    assert not is_transient_gemini_error(error)


def test_temporary_rate_limit_is_retryable():
    error = Exception(
        "429 Too Many Requests: rate limit exceeded"
    )

    assert not is_quota_exhausted_error(error)
    assert is_transient_gemini_error(error)


def test_service_unavailable_is_retryable():
    error = Exception(
        "503 Service Unavailable"
    )

    assert is_transient_gemini_error(error)


def test_invalid_request_is_not_retryable():
    error = Exception(
        "400 INVALID_ARGUMENT: invalid request"
    )

    assert not is_quota_exhausted_error(error)
    assert not is_transient_gemini_error(error)


def test_conceptual_question_prefers_semantic_search():
    assert is_semantic_query(
        "Where is path traversal prevented?"
    )


def test_exact_identifier_question_is_not_semantic():
    assert not is_semantic_query(
        "Where is _resolve_safe_path defined?"
    )


def test_dependency_question_is_not_semantic():
    assert not is_semantic_query(
        "What does repository.py depend on?"
    )
