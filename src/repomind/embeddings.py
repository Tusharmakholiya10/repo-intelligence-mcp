from __future__ import annotations

import os
import re
import time
from typing import Callable, Sequence, TypeVar

from google import genai
from google.genai import types


T = TypeVar("T")


class EmbeddingEngine:
    """Generate semantic embeddings using Gemini."""

    MODEL = "gemini-embedding-001"
    OUTPUT_DIMENSIONALITY = 768
    TASK_TYPE = "RETRIEVAL_DOCUMENT"

    MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 60.0

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Gemini embedding client.

        The API key is read from GEMINI_API_KEY when not
        explicitly provided.
        """

        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable "
                "is not set."
            )

        self.client = genai.Client(
            api_key=self.api_key
        )

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """
        Return True when the Gemini API indicates a 429/rate-limit
        condition.
        """

        status_code = getattr(error, "status_code", None)

        if status_code == 429:
            return True

        error_string = str(error).lower()

        return (
            "429" in error_string
            or "resource_exhausted" in error_string
            or "too many requests" in error_string
            or "quota exceeded" in error_string
        )

    @classmethod
    def _get_retry_delay(
        cls,
        error: Exception,
        attempt: int,
    ) -> float:
        """
        Determine how long to wait before retrying.

        Gemini often includes a server-provided retry delay such as:
        'retry in 44.5s' or 'retryDelay': '44s'.

        When no delay can be extracted, use exponential backoff.
        """

        error_string = str(error)

        patterns = (
            r"retry in\s+(\d+(?:\.\d+)?)s",
            r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"',
            r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                error_string,
                flags=re.IGNORECASE,
            )

            if match:
                return max(
                    1.0,
                    float(match.group(1)),
                )

        return max(
            cls.DEFAULT_RETRY_DELAY,
            2 ** attempt,
        )

    @classmethod
    def _with_retry(
        cls,
        operation: Callable[[], T],
    ) -> T:
        """
        Execute a Gemini operation with retry handling for 429 errors.

        Non-rate-limit errors are immediately re-raised.
        """

        for attempt in range(cls.MAX_RETRIES + 1):
            try:
                return operation()

            except Exception as error:

                if not cls._is_rate_limit_error(error):
                    raise

                if attempt >= cls.MAX_RETRIES:
                    raise

                delay = cls._get_retry_delay(
                    error,
                    attempt,
                )

                print(
                    "Gemini embedding quota/rate limit reached. "
                    f"Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{cls.MAX_RETRIES})..."
                )

                time.sleep(delay)

        raise RuntimeError(
            "Gemini embedding operation failed unexpectedly."
        )

    # ------------------------------------------------------------------
    # Single document embedding
    # ------------------------------------------------------------------

    def embed_document(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a repository document/chunk.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text to embed cannot be empty."
            )

        def operation():
            return self.client.models.embed_content(
                model=self.MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=self.TASK_TYPE,
                    output_dimensionality=(
                        self.OUTPUT_DIMENSIONALITY
                    ),
                ),
            )

        response = self._with_retry(operation)

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned no embedding."
            )

        embedding = response.embeddings[0]

        if not embedding.values:
            raise RuntimeError(
                "Gemini returned an empty embedding."
            )

        return list(embedding.values)

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a semantic search query.
        """

        if not text or not text.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        def operation():
            return self.client.models.embed_content(
                model=self.MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=(
                        self.OUTPUT_DIMENSIONALITY
                    ),
                ),
            )

        response = self._with_retry(operation)

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned no query embedding."
            )

        embedding = response.embeddings[0]

        if not embedding.values:
            raise RuntimeError(
                "Gemini returned an empty query embedding."
            )

        return list(embedding.values)

    # ------------------------------------------------------------------
    # Batch document embedding
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple repository chunks.

        Rate-limit errors are retried using Gemini's suggested delay
        when available.
        """

        if not texts:
            return []

        cleaned_texts = []

        for text in texts:
            if not text or not text.strip():
                raise ValueError(
                    "Document text cannot be empty."
                )

            cleaned_texts.append(text)

        def operation():
            return self.client.models.embed_content(
                model=self.MODEL,
                contents=cleaned_texts,
                config=types.EmbedContentConfig(
                    task_type=self.TASK_TYPE,
                    output_dimensionality=(
                        self.OUTPUT_DIMENSIONALITY
                    ),
                ),
            )

        response = self._with_retry(operation)

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned no embeddings."
            )

        if len(response.embeddings) != len(
            cleaned_texts
        ):
            raise RuntimeError(
                "Gemini returned an unexpected number "
                "of embeddings."
            )

        embeddings = []

        for embedding in response.embeddings:
            if not embedding.values:
                raise RuntimeError(
                    "Gemini returned an empty embedding."
                )

            embeddings.append(
                list(embedding.values)
            )

        return embeddings

    def close(self):
        """Close the Gemini client."""
        self.client.close()