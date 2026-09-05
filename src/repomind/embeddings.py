from __future__ import annotations

import os
from typing import Sequence

from google import genai
from google.genai import types


class EmbeddingEngine:
    """Generate semantic embeddings using Gemini."""

    MODEL = "gemini-embedding-001"
    OUTPUT_DIMENSIONALITY = 768
    TASK_TYPE = "RETRIEVAL_DOCUMENT"

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

    def embed_document(self, text: str) -> list[float]:
        """
        Generate an embedding for a repository document/chunk.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text to embed cannot be empty."
            )

        response = self.client.models.embed_content(
            model=self.MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=self.TASK_TYPE,
                output_dimensionality=(
                    self.OUTPUT_DIMENSIONALITY
                ),
            ),
        )

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

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a semantic search query.
        """

        if not text or not text.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        response = self.client.models.embed_content(
            model=self.MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=(
                    self.OUTPUT_DIMENSIONALITY
                ),
            ),
        )

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

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple repository chunks.
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

        response = self.client.models.embed_content(
            model=self.MODEL,
            contents=cleaned_texts,
            config=types.EmbedContentConfig(
                task_type=self.TASK_TYPE,
                output_dimensionality=(
                    self.OUTPUT_DIMENSIONALITY
                ),
            ),
        )

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