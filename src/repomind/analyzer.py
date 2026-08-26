import ast
from pathlib import Path


class PythonAnalyzer:
    """Analyze Python source code using the AST."""

    def __init__(self, repository_root: Path):
        self.repository_root = repository_root.resolve()

    def analyze_file(self, relative_path: str) -> list[dict]:
        """
        Extract structured symbols from a Python file.
        """

        path = self._resolve_python_file(relative_path)

        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                "File is not valid UTF-8."
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(
                f"Could not parse {relative_path}: {exc}"
            )

        symbols = []

        for node in tree.body:
            self._extract_node(
                node=node,
                symbols=symbols,
                parent=None,
            )

        return symbols

    def _extract_node(
        self,
        node: ast.AST,
        symbols: list[dict],
        parent: str | None,
    ) -> None:
        """Recursively extract symbols while preserving scope."""

        if isinstance(node, ast.ClassDef):

            symbols.append({
                "type": "class",
                "name": node.name,
                "qualified_name": (
                    f"{parent}.{node.name}"
                    if parent
                    else node.name
                ),
                "line": node.lineno,
                "end_line": node.end_lineno,
            })

            current_scope = (
                f"{parent}.{node.name}"
                if parent
                else node.name
            )

            for child in node.body:
                self._extract_node(
                    child,
                    symbols,
                    current_scope,
                )

            return

        if isinstance(node, ast.FunctionDef):

            qualified_name = (
                f"{parent}.{node.name}"
                if parent
                else node.name
            )

            symbols.append({
                "type": "method" if parent else "function",
                "name": node.name,
                "qualified_name": qualified_name,
                "line": node.lineno,
                "end_line": node.end_lineno,
            })

            return

        if isinstance(node, ast.AsyncFunctionDef):

            qualified_name = (
                f"{parent}.{node.name}"
                if parent
                else node.name
            )

            symbols.append({
                "type": (
                    "async_method"
                    if parent
                    else "async_function"
                ),
                "name": node.name,
                "qualified_name": qualified_name,
                "line": node.lineno,
                "end_line": node.end_lineno,
            })

            return

        if isinstance(node, ast.Import):

            for alias in node.names:

                symbols.append({
                    "type": "import",
                    "name": alias.name,
                    "qualified_name": alias.asname or alias.name,
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                })

            return

        if isinstance(node, ast.ImportFrom):

            module = node.module or ""

            for alias in node.names:

                full_name = (
                    f"{module}.{alias.name}"
                    if module
                    else alias.name
                )

                symbols.append({
                    "type": "import",
                    "name": full_name,
                    "qualified_name": (
                        alias.asname or alias.name
                    ),
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                })

            return

    def _resolve_python_file(
        self,
        relative_path: str,
    ) -> Path:
        """Safely resolve a Python file inside the repository."""

        requested_path = Path(relative_path)

        if requested_path.is_absolute():
            raise ValueError(
                "Absolute paths are not allowed."
            )

        path = (
            self.repository_root / requested_path
        ).resolve()

        try:
            path.relative_to(self.repository_root)
        except ValueError:
            raise ValueError(
                "Access denied: file is outside "
                "the repository."
            )

        if not path.exists():
            raise ValueError(
                f"File not found: {relative_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {relative_path}"
            )

        if path.suffix.lower() != ".py":
            raise ValueError(
                "AST analysis currently supports "
                "Python files only."
            )

        return path

        def search_symbols(
            self,
            query: str,
            max_results: int = 100,
        ) -> list[dict]:
            """
            Search for symbols across all Python files
            in the repository.
            """

            if not query.strip():
                raise ValueError(
                    "Symbol search query cannot be empty."
                )

            results = []

            ignored_directories = {
                ".git",
                "venv",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                "node_modules",
                "dist",
                "build",
            }

            for path in sorted(self.repository_root.rglob("*.py")):

                if any(
                    part in ignored_directories
                    for part in path.parts
                ):
                    continue

                relative_path = path.relative_to(
                    self.repository_root
                )

                try:
                    symbols = self.analyze_file(
                        str(relative_path)
                    )
                except ValueError:
                    continue

                for symbol in symbols:

                    if query.lower() in (
                        symbol["name"].lower()
                    ) or query.lower() in (
                        symbol["qualified_name"].lower()
                    ):

                        result = {
                            "file": str(relative_path),
                            **symbol,
                        }

                        results.append(result)

                        if len(results) >= max_results:
                            return results

            return results