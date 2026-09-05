from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeChunk:
    """A searchable chunk of repository source code."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    symbol_name: str | None = None
    symbol_type: str | None = None


class CodeChunker:
    """
    Convert source files into semantically meaningful chunks.

    Python symbols discovered by PythonAnalyzer are preferred.
    Large symbols are split into smaller line-based chunks.
    """

    DEFAULT_MAX_LINES = 80
    DEFAULT_MAX_CHARS = 6000

    def __init__(
        self,
        max_lines: int = DEFAULT_MAX_LINES,
        max_chars: int = DEFAULT_MAX_CHARS,
    ):
        if max_lines < 1:
            raise ValueError(
                "max_lines must be at least 1."
            )

        if max_chars < 1:
            raise ValueError(
                "max_chars must be at least 1."
            )

        self.max_lines = max_lines
        self.max_chars = max_chars

    def chunk_python_file(
        self,
        file_path: str,
        source_text: str,
        symbols: list[dict],
    ) -> list[CodeChunk]:
        """
        Create semantic chunks from a Python source file.

        Each chunk is associated with a discovered symbol when
        possible. Large symbol bodies are split into smaller pieces.
        """

        if not file_path or not file_path.strip():
            raise ValueError(
                "file_path cannot be empty."
            )

        if not source_text:
            return []

        lines = source_text.splitlines()

        if not lines:
            return []

        chunks = []
        seen_ranges = set()

        normalized_symbols = self._normalize_symbols(
            symbols,
            len(lines),
        )

        for symbol in normalized_symbols:

            symbol_name = symbol.get(
                "qualified_name"
            ) or symbol.get("name")

            symbol_type = symbol.get(
                "type"
            )

            start_line = symbol["line"]
            end_line = symbol["end_line"]

            symbol_lines = lines[
                start_line - 1:end_line
            ]

            if not symbol_lines:
                continue

            for chunk in self._split_lines(
                symbol_lines,
                start_line,
            ):

                chunk_start = chunk["start_line"]
                chunk_end = chunk["end_line"]

                range_key = (
                    file_path,
                    chunk_start,
                    chunk_end,
                )

                if range_key in seen_ranges:
                    continue

                seen_ranges.add(
                    range_key
                )

                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        start_line=chunk_start,
                        end_line=chunk_end,
                        content=chunk["content"],
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                    )
                )

        # If no symbols were found, fall back to line-based chunks.
        if not chunks:

            for chunk in self._split_lines(
                lines,
                1,
            ):

                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        start_line=chunk["start_line"],
                        end_line=chunk["end_line"],
                        content=chunk["content"],
                    )
                )

        return chunks

    def _normalize_symbols(
        self,
        symbols: list[dict],
        line_count: int,
    ) -> list[dict]:
        """Validate and normalize analyzer symbol ranges."""

        normalized = []

        for symbol in symbols:

            if "line" not in symbol:
                continue

            start_line = symbol["line"]

            end_line = symbol.get(
                "end_line",
                start_line,
            )

            if not isinstance(
                start_line,
                int,
            ):

                continue

            if not isinstance(
                end_line,
                int,
            ):

                continue

            start_line = max(
                1,
                start_line,
            )

            end_line = min(
                line_count,
                end_line,
            )

            if start_line > end_line:
                continue

            normalized_symbol = dict(
                symbol
            )

            normalized_symbol["line"] = (
                start_line
            )

            normalized_symbol["end_line"] = (
                end_line
            )

            normalized.append(
                normalized_symbol
            )

        normalized.sort(
            key=lambda symbol: (
                symbol["line"],
                symbol["end_line"],
            )
        )

        return normalized

    def _split_lines(
        self,
        lines: list[str],
        starting_line: int,
    ) -> list[dict]:
        """
        Split lines according to line and character limits.

        This keeps chunks small enough for efficient embedding.
        """

        chunks = []

        current_lines = []
        current_chars = 0
        current_start = starting_line

        for offset, line in enumerate(
            lines
        ):

            line_number = (
                starting_line + offset
            )

            line_chars = len(line) + 1

            would_exceed_lines = (
                len(current_lines)
                >= self.max_lines
            )

            would_exceed_chars = (
                current_lines
                and
                current_chars + line_chars
                > self.max_chars
            )

            if (
                would_exceed_lines
                or would_exceed_chars
            ):

                chunks.append(
                    {
                        "start_line": current_start,
                        "end_line": (
                            line_number - 1
                        ),
                        "content": "\n".join(
                            current_lines
                        ),
                    }
                )

                current_lines = []
                current_chars = 0
                current_start = line_number

            current_lines.append(
                line
            )

            current_chars += line_chars

        if current_lines:

            chunks.append(
                {
                    "start_line": current_start,
                    "end_line": (
                        starting_line
                        + len(lines)
                        - 1
                    ),
                    "content": "\n".join(
                        current_lines
                    ),
                }
            )

        return chunks