from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MAX_CHARS = 12000
DEFAULT_CONTEXT_LINES = 12
DEFAULT_MAX_RELATED = 5
DEFAULT_MAX_USAGES = 5
DEFAULT_MAX_DEPENDENCIES = 5


@dataclass(frozen=True)
class PackedContext:
    """Compact repository context prepared for an AI agent."""

    path: str
    symbol_name: str | None
    content: str
    total_characters: int
    truncated: bool


def _find_symbol(
    symbols: list[dict],
    symbol_name: str | None,
) -> dict | None:
    if not symbol_name:
        return None

    normalized = symbol_name.strip()

    for symbol in symbols:
        if (
            symbol.get("qualified_name")
            == normalized
        ):
            return symbol

        if symbol.get("name") == normalized:
            return symbol

    short_name = normalized.rsplit(
        ".",
        1,
    )[-1]

    for symbol in symbols:
        if symbol.get("name") == short_name:
            return symbol

    return None


def _extract_symbol_context(
    source: str,
    symbol: dict | None,
    context_lines: int,
) -> tuple[str, bool]:
    """
    Extract a source window around the primary symbol.

    Returns the rendered text along with a flag indicating whether the
    window is narrower than the full source (i.e. some of the source
    was left out of the window itself, independent of any later
    max_chars truncation).
    """

    lines = source.splitlines()

    if not lines:
        return "", False

    if symbol is None:
        end = min(
            len(lines),
            max(1, context_lines * 3),
        )

        text = "\n".join(
            f"{index + 1}: {lines[index]}"
            for index in range(end)
        )

        return text, end < len(lines)

    start_line = max(
        1,
        int(symbol["line"]) - context_lines,
    )

    end_line = min(
        len(lines),
        int(symbol["end_line"]) + context_lines,
    )

    text = "\n".join(
        f"{index}: {lines[index - 1]}"
        for index in range(
            start_line,
            end_line + 1,
        )
    )

    window_truncated = (
        start_line > 1
        or end_line < len(lines)
    )

    return text, window_truncated


def _format_related_symbols(
    symbols: list[dict],
    primary: dict | None,
    max_related: int,
) -> str:
    primary_name = (
        primary.get("qualified_name")
        if primary
        else None
    )

    related = []

    for symbol in symbols:
        if (
            symbol.get("qualified_name")
            == primary_name
        ):
            continue

        related.append(
            "- "
            f"{symbol.get('type', 'symbol')}: "
            f"{symbol.get('qualified_name', symbol.get('name'))} "
            f"(lines "
            f"{symbol.get('line')}-"
            f"{symbol.get('end_line')})"
        )

        if len(related) >= max_related:
            break

    return "\n".join(related)


def _format_usages(
    usages: list[dict],
) -> str:
    if not usages:
        return "No indexed usages found."

    return "\n".join(
        (
            f"- {usage['path']}:"
            f"{usage['line']} "
            f"[{usage['reference_type']}] "
            f"{usage['symbol_name']}"
        )
        for usage in usages
    )


def _format_dependencies(
    dependencies: list[dict],
) -> str:
    if not dependencies:
        return "No local dependencies found."

    return "\n".join(
        (
            f"- {dependency['path']} "
            f"[{dependency['dependency_type']}] "
            f"line {dependency['line']}"
        )
        for dependency in dependencies
    )


def pack_context(
    *,
    path: str,
    symbol_name: str | None,
    source: str,
    symbols: list[dict],
    usages: list[dict],
    dependencies: list[dict],
    max_chars: int = DEFAULT_MAX_CHARS,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    max_related: int = DEFAULT_MAX_RELATED,
) -> PackedContext:
    """
    Build a compact, structured context bundle.

    The result is deterministic and does not call an LLM.
    """

    if max_chars < 500:
        raise ValueError(
            "max_chars must be at least 500."
        )

    primary = _find_symbol(
        symbols,
        symbol_name,
    )

    resolved_symbol = (
        primary.get("qualified_name")
        if primary
        else symbol_name
    )

    source_context, source_window_truncated = (
        _extract_symbol_context(
            source,
            primary,
            context_lines,
        )
    )

    sections = [
        (
            "PRIMARY",
            "\n".join(
                [
                    f"Path: {path}",
                    (
                        f"Symbol: {resolved_symbol}"
                        if resolved_symbol
                        else "Symbol: <file-level context>"
                    ),
                    (
                        f"Type: {primary['type']}"
                        if primary
                        else "Type: file"
                    ),
                    (
                        f"Lines: {primary['line']}-"
                        f"{primary['end_line']}"
                        if primary
                        else "Lines: unavailable"
                    ),
                ]
            ),
        ),
        (
            "SOURCE",
            source_context,
        ),
        (
            "RELATED SYMBOLS",
            _format_related_symbols(
                symbols,
                primary,
                max_related,
            ),
        ),
        (
            "USAGES",
            _format_usages(
                usages
            ),
        ),
        (
            "DEPENDENCIES",
            _format_dependencies(
                dependencies
            ),
        ),
    ]

    rendered = []

    for title, body in sections:
        section = (
            f"=== {title} ===\n"
            f"{body.strip()}\n"
        )

        current_length = sum(
            len(item)
            for item in rendered
        )

        remaining = (
            max_chars
            - current_length
        )

        if remaining <= 0:
            break

        if len(section) > remaining:
            section = section[
                :remaining
            ]

        rendered.append(section)

        if len(section) < len(
            f"=== {title} ===\n"
        ):
            break

    content = "".join(rendered)

    max_chars_truncated = (
        len(content)
        < sum(
            len(body) + len(title) + 8
            for title, body in sections
        )
    )

    truncated = (
        source_window_truncated
        or max_chars_truncated
    )

    return PackedContext(
        path=path,
        symbol_name=resolved_symbol,
        content=content.rstrip(),
        total_characters=len(content),
        truncated=truncated,
    )