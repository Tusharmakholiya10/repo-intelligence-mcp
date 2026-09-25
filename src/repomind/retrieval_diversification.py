from __future__ import annotations


def diversify_results(
    results: list[dict],
    max_results: int,
) -> list[dict]:
    """
    Remove duplicate symbol-level results while preserving ranking order.

    The highest-ranked result for each path/symbol combination is retained.
    Results without a symbol name are treated as unique chunks.
    """

    diversified = []
    seen_symbols = set()

    for result in results:
        symbol_name = result.get("symbol_name")
        path = result.get("path")

        if symbol_name:
            key = (
                path,
                symbol_name,
            )

            if key in seen_symbols:
                continue

            seen_symbols.add(key)

        diversified.append(result)

        if len(diversified) >= max_results:
            break

    return diversified