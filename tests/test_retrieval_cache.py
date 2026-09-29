from __future__ import annotations

import threading
import time

import pytest

from repomind.retrieval_cache import RetrievalCache, TTLCache


def test_ttl_cache_miss_and_hit():
    cache = TTLCache(max_size=2, ttl_seconds=60)

    assert cache.get("missing") is None

    cache.set("key", {"value": 123})

    assert cache.get("key") == {"value": 123}

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_ttl_cache_returns_deep_copy():
    cache = TTLCache(max_size=2, ttl_seconds=60)

    original = {"items": [1, 2, 3]}
    cache.set("key", original)

    result = cache.get("key")
    result["items"].append(4)

    assert cache.get("key") == {"items": [1, 2, 3]}


def test_ttl_cache_expires_entries():
    cache = TTLCache(max_size=2, ttl_seconds=0.01)

    cache.set("key", "value")

    assert cache.get("key") == "value"

    time.sleep(0.03)

    assert cache.get("key") is None

    stats = cache.stats()
    assert stats["misses"] == 1


def test_ttl_cache_lru_eviction():
    cache = TTLCache(max_size=2, ttl_seconds=60)

    cache.set("a", 1)
    cache.set("b", 2)

    # Refresh "a", making "b" the least recently used entry.
    assert cache.get("a") == 1

    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3

    stats = cache.stats()
    assert stats["evictions"] == 1


def test_ttl_cache_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        TTLCache(max_size=0, ttl_seconds=60)

    with pytest.raises(ValueError):
        TTLCache(max_size=1, ttl_seconds=0)

    with pytest.raises(ValueError):
        TTLCache(max_size=1, ttl_seconds=-1)


def test_retrieval_cache_normalizes_queries():
    cache = RetrievalCache()

    assert (
        cache.normalize_query("  Hello   WORLD  ")
        == "hello world"
    )


def test_embedding_keys_are_normalized():
    cache = RetrievalCache()

    key_one = cache.build_embedding_key(
        "/repo",
        "  Semantic   Search ",
    )
    key_two = cache.build_embedding_key(
        "/repo",
        "semantic search",
    )

    assert key_one == key_two


def test_result_keys_capture_search_parameters():
    cache = RetrievalCache()

    base = cache.build_result_key(
        "/repo",
        "semantic search",
        "semantic_search",
        5,
        0.20,
    )

    different_intent = cache.build_result_key(
        "/repo",
        "semantic search",
        "architecture",
        5,
        0.20,
    )

    different_limit = cache.build_result_key(
        "/repo",
        "semantic search",
        "semantic_search",
        10,
        0.20,
    )

    different_threshold = cache.build_result_key(
        "/repo",
        "semantic search",
        "semantic_search",
        5,
        0.30,
    )

    assert base != different_intent
    assert base != different_limit
    assert base != different_threshold


def test_retrieval_cache_clear():
    cache = RetrievalCache()

    cache.set_embedding("embedding-key", [1.0, 2.0])
    cache.set_results("result-key", [{"score": 0.9}])

    assert cache.get_embedding("embedding-key") == [1.0, 2.0]
    assert cache.get_results("result-key") == [{"score": 0.9}]

    cache.clear()

    assert cache.get_embedding("embedding-key") is None
    assert cache.get_results("result-key") is None


def test_retrieval_cache_stats():
    cache = RetrievalCache()

    cache.set_embedding("embedding-key", [1.0])
    cache.get_embedding("embedding-key")
    cache.get_embedding("missing")

    cache.set_results("result-key", [{"score": 0.8}])
    cache.get_results("result-key")
    cache.get_results("missing")

    stats = cache.stats()

    assert "embedding" in stats
    assert "results" in stats

    assert stats["embedding"]["hits"] == 1
    assert stats["embedding"]["misses"] == 1

    assert stats["results"]["hits"] == 1
    assert stats["results"]["misses"] == 1


def test_ttl_cache_concurrent_access():
    cache = TTLCache(max_size=32, ttl_seconds=60)

    errors: list[Exception] = []

    def worker(worker_id: int):
        try:
            for index in range(100):
                key = f"{worker_id}-{index % 10}"
                cache.set(key, {"worker": worker_id, "index": index})
                cache.get(key)
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(
            target=worker,
            args=(worker_id,),
        )
        for worker_id in range(8)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert not errors