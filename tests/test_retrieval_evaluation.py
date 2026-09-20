from evaluation.run_retrieval_eval import (
    calculate_metrics,
    check_thresholds,
    result_matches_expected,
)


def test_result_matches_expected_path():
    result = {
        "path": "src/repomind/repository.py",
        "symbol_name": "_resolve_safe_path",
    }

    assert result_matches_expected(
        result,
        "src/repomind/repository.py",
        [],
    )


def test_result_matches_expected_symbol():
    result = {
        "path": "src/repomind/repository.py",
        "symbol_name": "_resolve_safe_path",
    }

    assert result_matches_expected(
        result,
        "src/repomind/repository.py",
        ["_resolve_safe_path"],
    )


def test_result_rejects_wrong_path():
    result = {
        "path": "src/repomind/indexer.py",
        "symbol_name": "_resolve_safe_path",
    }

    assert not result_matches_expected(
        result,
        "src/repomind/repository.py",
        ["_resolve_safe_path"],
    )


def test_metrics_calculation():
    results = [
        {
            "hit_at_1": True,
            "hit_at_3": True,
            "reciprocal_rank": 1.0,
        },
        {
            "hit_at_1": False,
            "hit_at_3": True,
            "reciprocal_rank": 0.5,
        },
        {
            "hit_at_1": False,
            "hit_at_3": False,
            "reciprocal_rank": 0.0,
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics["queries_evaluated"] == 3
    assert metrics["hit_at_1"] == 1 / 3
    assert metrics["hit_at_3"] == 2 / 3
    assert metrics["mrr"] == 0.5


def test_regression_threshold_passes():
    metrics = {
        "hit_at_1": 0.50,
        "hit_at_3": 0.90,
        "mrr": 0.70,
    }

    config = {
        "minimum_hit_at_1": 0.40,
        "minimum_hit_at_3": 0.80,
        "minimum_mrr": 0.55,
    }

    passed, failures = check_thresholds(
        metrics,
        config,
    )

    assert passed
    assert failures == []


def test_regression_threshold_fails():
    metrics = {
        "hit_at_1": 0.30,
        "hit_at_3": 0.70,
        "mrr": 0.40,
    }

    config = {
        "minimum_hit_at_1": 0.40,
        "minimum_hit_at_3": 0.80,
        "minimum_mrr": 0.55,
    }

    passed, failures = check_thresholds(
        metrics,
        config,
    )

    assert not passed
    assert len(failures) == 3