from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    created_at: float


class TTLCache:
    """Small thread-safe TTL + LRU cache."""

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

        self._entries: OrderedDict[str, CacheEntry] = (
            OrderedDict()
        )
        self._lock = threading.RLock()

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
    ) -> Any | None:
        with self._lock:
            now = time.monotonic()

            entry = self._entries.get(key)

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

            # Move the accessed item to the end so that
            # the first item remains the least-recently-used.
            self._entries.move_to_end(key)

            self.hits += 1

            # Return a deep copy so callers cannot mutate
            # the object stored inside the cache.
            return copy.deepcopy(entry.value)

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        with self._lock:
            now = time.monotonic()

            self._entries[key] = CacheEntry(
                value=copy.deepcopy(value),
                created_at=now,
            )

            self._entries.move_to_end(key)

            # Remove the least-recently-used entries
            # until the cache is back within its limit.
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
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
        """
        Normalize a query for stable cache-key generation.

        Leading/trailing whitespace is removed,
        repeated whitespace is collapsed, and the
        query is converted to lowercase.
        """
        return " ".join(
            query.strip().lower().split()
        )

    def build_embedding_key(
        self,
        repository_root: str,
        query: str,
    ) -> str:
        """
        Build a stable cache key for query embeddings.
        """
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
        """
        Build a stable cache key for semantic-search results.

        Search configuration is part of the key so that
        different retrieval settings cannot reuse the
        wrong cached result set.
        """
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
    ) -> list[float] | None:
        """
        Retrieve a cached query embedding.
        """
        return self.embedding_cache.get(key)

    def set_embedding(
        self,
        key: str,
        embedding: list[float],
    ) -> None:
        """
        Store a query embedding in the cache.
        """
        self.embedding_cache.set(
            key,
            embedding,
        )

    def get_results(
        self,
        key: str,
    ) -> list[dict] | None:
        """
        Retrieve cached semantic-search results.
        """
        return self.result_cache.get(key)

    def set_results(
        self,
        key: str,
        results: list[dict],
    ) -> None:
        """
        Store semantic-search results in the cache.
        """
        self.result_cache.set(
            key,
            results,
        )

    def clear(self) -> None:
        """
        Clear both embedding and result caches.
        """
        self.embedding_cache.clear()
        self.result_cache.clear()

    def stats(self) -> dict[str, dict[str, int]]:
        """
        Return statistics for both cache layers.
        """
        return {
            "embedding": self.embedding_cache.stats(),
            "results": self.result_cache.stats(),
        }