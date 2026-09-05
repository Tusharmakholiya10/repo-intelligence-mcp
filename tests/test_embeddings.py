import os

import pytest

from repomind.embeddings import EmbeddingEngine


def test_embedding_engine_requires_api_key(
    monkeypatch,
):
    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False,
    )

    with pytest.raises(ValueError):
        EmbeddingEngine()


def test_embedding_engine_accepts_explicit_api_key():
    engine = EmbeddingEngine(
        api_key="test-key"
    )

    assert engine.api_key == "test-key"

    engine.close()