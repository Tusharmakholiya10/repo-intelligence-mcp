from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    created_at: float


class TTLCache:
    """Small thread-safe-independent TTL + LRU cache."""

    def __init__(
        self,
        max_size: int,
        ttl_seconds: float,
    ):
        if max_size < 1:
            raise ValueError(
                "max_size must be at least 1."
            )

        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than 0."
            )

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        self._entries = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _is_expired(
        self,
        entry: CacheEntry,
        now: float,
    ) -> bool:
        return (
            now - entry.created_at
            >= self.ttl_seconds
        )

    def get(
        self,
        key: str,
    ):
        now = time.monotonic()

        entry = self._entries.get(
            key
        )

        if entry is None:
            self.misses += 1
            return None

        if self._is_expired(
            entry,
            now,
        ):
            del self._entries[key]
            self.misses += 1
            return None

        self._entries.move_to_end(
            key
        )

        self.hits += 1

        return copy.deepcopy(
            entry.value
        )

    def set(
        self,
        key: str,
        value,
    ) -> None:
        now = time.monotonic()

        self._entries[key] = CacheEntry(
            value=copy.deepcopy(value),
            created_at=now,
        )

        self._entries.move_to_end(
            key
        )

        while (
            len(self._entries)
            > self.max_size
        ):
            self._entries.popitem(
                last=False
            )
            self.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict[str, int]:
        return {
            "size": len(self._entries),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class RetrievalCache:
    """Cache semantic retrieval inputs and outputs."""

    EMBEDDING_MAX_SIZE = 128
    RESULT_MAX_SIZE = 64
    TTL_SECONDS = 600.0

    def __init__(self):
        self.embedding_cache = TTLCache(
            max_size=self.EMBEDDING_MAX_SIZE,
            ttl_seconds=self.TTL_SECONDS,
        )

        self.result_cache = TTLCache(
            max_size=self.RESULT_MAX_SIZE,
            ttl_seconds=self.TTL_SECONDS,
        )

    @staticmethod
    def normalize_query(
        query: str,
    ) -> str:
        return " ".join(
            query.strip().lower().split()
        )

    def build_embedding_key(
        self,
        repository_root: str,
        query: str,
    ) -> str:
        normalized = self.normalize_query(
            query
        )

        return (
            f"embedding|"
            f"{repository_root}|"
            f"{normalized}"
        )

    def build_result_key(
        self,
        repository_root: str,
        query: str,
        query_intent: str,
        max_results: int,
        min_similarity: float,
    ) -> str:
        normalized = self.normalize_query(
            query
        )

        return (
            f"result|"
            f"{repository_root}|"
            f"{normalized}|"
            f"{query_intent}|"
            f"{max_results}|"
            f"{min_similarity:.6f}"
        )

    def get_embedding(
        self,
        key: str,
    ):
        return self.embedding_cache.get(
            key
        )

    def set_embedding(
        self,
        key: str,
        embedding: list[float],
    ) -> None:
        self.embedding_cache.set(
            key,
            embedding,
        )

    def get_results(
        self,
        key: str,
    ):
        return self.result_cache.get(
            key
        )

    def set_results(
        self,
        key: str,
        results: list[dict],
    ) -> None:
        self.result_cache.set(
            key,
            results,
        )

    def clear(self) -> None:
        self.embedding_cache.clear()
        self.result_cache.clear()

    def stats(self) -> dict:
        return {
            "embedding": (
                self.embedding_cache.stats()
            ),
            "results": (
                self.result_cache.stats()
            ),
        }