from __future__ import annotations
import argparse

import json
import sys
from pathlib import Path


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from repomind.embeddings import EmbeddingEngine
from repomind.indexer import CodeIndexer


DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_dataset.json"
)

DEFAULT_TOP_K = 5
RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "latest.json"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_config.json"
)

def load_dataset() -> list[dict]:
    """Load retrieval evaluation questions."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    return dataset

def load_config() -> dict:
    """Load retrieval evaluation thresholds."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation config not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Evaluation config must contain a JSON object."
        )

    return config

def normalize_path(path: str) -> str:
    """Normalize repository paths for comparison."""

    return path.replace("\\", "/").lower()


def result_matches_expected(
    result: dict,
    expected_path: str,
    expected_symbols: list[str],
) -> bool:
    """
    Determine whether one semantic-search result
    satisfies the expected file/symbol criteria.
    """

    result_path = normalize_path(
        str(result.get("path", ""))
    )

    expected_path_normalized = normalize_path(
        expected_path
    )

    path_match = (
        result_path == expected_path_normalized
        or result_path.endswith(
            expected_path_normalized
        )
    )

    if not path_match:
        return False

    if not expected_symbols:
        return True

    symbol_name = (
        str(result.get("symbol_name") or "")
        .lower()
    )

    return any(
        symbol.lower() in symbol_name
        for symbol in expected_symbols
    )


def evaluate_query(
    indexer: CodeIndexer,
    embedding_engine: EmbeddingEngine,
    item: dict,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Evaluate one retrieval query."""

    question = item["question"]
    expected_path = item["expected_path"]
    expected_symbols = item.get(
        "expected_symbols",
        [],
    )

    query_embedding = (
        embedding_engine.embed_query(
            question
        )
    )

    results = indexer.semantic_search(
        query_embedding=query_embedding,
        query_text=question,
        max_results=top_k,
        min_similarity=0.0,
    )

    hit_rank = None

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result_matches_expected(
            result,
            expected_path,
            expected_symbols,
        ):
            hit_rank = rank
            break

    hit_at_1 = (
        hit_rank == 1
    )

    hit_at_3 = (
        hit_rank is not None
        and hit_rank <= 3
    )

    reciprocal_rank = (
        1.0 / hit_rank
        if hit_rank is not None
        else 0.0
    )

    return {
        "id": item["id"],
        "question": question,
        "expected_path": expected_path,
        "expected_symbols": expected_symbols,
        "hit_rank": hit_rank,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "reciprocal_rank": reciprocal_rank,
        "results": results,
    }

def calculate_metrics(
    results: list[dict],
) -> dict:
    """Calculate aggregate retrieval metrics."""

    total = len(results)

    if total == 0:
        return {
            "queries_evaluated": 0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "mrr": 0.0,
        }

    hit_at_1 = sum(
        result["hit_at_1"]
        for result in results
    )

    hit_at_3 = sum(
        result["hit_at_3"]
        for result in results
    )

    mrr = sum(
        result["reciprocal_rank"]
        for result in results
    ) / total

    return {
        "queries_evaluated": total,
        "hit_at_1": hit_at_1 / total,
        "hit_at_3": hit_at_3 / total,
        "mrr": mrr,
    }

def check_thresholds(
    metrics: dict,
    config: dict,
) -> tuple[bool, list[str]]:
    """Check metrics against configured regression floors."""

    checks = {
        "hit_at_1": float(
            config["minimum_hit_at_1"]
        ),
        "hit_at_3": float(
            config["minimum_hit_at_3"]
        ),
        "mrr": float(
            config["minimum_mrr"]
        ),
    }

    failures = []

    for metric_name, minimum in checks.items():
        actual = float(metrics[metric_name])

        if actual < minimum:
            failures.append(
                f"{metric_name}={actual:.4f} "
                f"is below minimum {minimum:.4f}"
            )

    return not failures, failures

def save_report(
    metrics: dict,
    results: list[dict],
) -> None:
    """Save the latest evaluation report."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "metrics": metrics,
        "cases": results,
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        f"\nEvaluation report saved to: "
        f"{RESULTS_PATH}"
    )

def print_result(result: dict) -> None:
    """Print one evaluation result."""

    print()
    print("-" * 70)
    print(
        f"Question: {result['question']}"
    )
    print(
        f"Expected: {result['expected_path']}"
    )
    print(
        "Expected symbols: "
        + ", ".join(
            result["expected_symbols"]
        )
    )

    if result["hit_rank"] is None:
        print("Result: MISS")
    else:
        print(
            f"Result: HIT at rank "
            f"{result['hit_rank']}"
        )

    print("Top results:")

    for rank, item in enumerate(
        result["results"][:3],
        start=1,
    ):
        print(
            f"  {rank}. "
            f"{item['path']}:"
            f"{item['start_line']}-"
            f"{item['end_line']}"
        )

        symbol_name = item.get(
            "symbol_name"
        )

        if symbol_name:
            print(
                f"     Symbol: {symbol_name}"
            )

        print(
            f"     Score: "
            f"{item.get('score', 0.0):.4f}"
        )


def print_summary(
    results: list[dict],
) -> dict:
    """Print and return aggregate retrieval metrics."""

    metrics = calculate_metrics(results)

    print()
    print("=" * 70)
    print("RepoMind Retrieval Evaluation")
    print("=" * 70)

    print(
        f"Queries evaluated: "
        f"{metrics['queries_evaluated']}"
    )

    print(
        f"Hit@1: "
        f"{metrics['hit_at_1']:.1%}"
    )

    print(
        f"Hit@3: "
        f"{metrics['hit_at_3']:.1%}"
    )

    print(
        f"MRR: "
        f"{metrics['mrr']:.4f}"
    )

    print("=" * 70)

    return metrics

def main() -> int:
    """Run the retrieval benchmark."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the RepoMind retrieval benchmark."
        )
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to evaluation/results/latest.json.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check results against regression thresholds.",
    )

    args = parser.parse_args()

    dataset = load_dataset()

    indexer = CodeIndexer(
        PROJECT_ROOT
    )

    embedding_engine = EmbeddingEngine()

    results = []

    try:
        for item in dataset:

            result = evaluate_query(
                indexer,
                embedding_engine,
                item,
            )

            results.append(result)

            print_result(result)

    finally:
        embedding_engine.close()

    metrics = print_summary(results)

    if args.save:
        save_report(
            metrics,
            results,
        )

    if args.check:
        config = load_config()

        passed, failures = check_thresholds(
            metrics,
            config,
        )

        print()
        print(
            "Retrieval regression check"
        )
        print("-" * 30)

        for metric_name in (
            "hit_at_1",
            "hit_at_3",
            "mrr",
        ):
            minimum = float(
                config[
                    f"minimum_{metric_name}"
                ]
            )

            actual = metrics[metric_name]

            status = (
                "PASS"
                if actual >= minimum
                else "FAIL"
            )

            print(
                f"{metric_name.upper():7} : "
                f"{actual:.4f} "
                f"{status} "
                f"(minimum {minimum:.4f})"
            )

        if not passed:
            print()
            print(
                "Evaluation failed."
            )

            for failure in failures:
                print(
                    f"  - {failure}"
                )

            return 1

        print()
        print(
            "Evaluation passed."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )