import ast
from pathlib import Path


class PythonAnalyzer:
    """Analyze Python source code using the AST."""

    def __init__(self, repository_root: Path):
        self.repository_root = repository_root.resolve()

    def analyze_file(self, relative_path: str) -> list[dict]:
        """
        Extract symbols from a Python file.

        Returns classes, functions, methods and imports.
        """

        path = (self.repository_root / relative_path).resolve()

        try:
            path.relative_to(self.repository_root)
        except ValueError:
            raise ValueError(
                "Access denied: file is outside the repository."
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
                "AST analysis currently supports Python files only."
            )

        try:
            source = path.read_text(
                encoding="utf-8"
            )
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

        for node in ast.walk(tree):

            if isinstance(node, ast.ClassDef):

                symbols.append({
                    "type": "class",
                    "name": node.name,
                    "line": node.lineno,
                })

            elif isinstance(node, ast.FunctionDef):

                symbols.append({
                    "type": "function",
                    "name": node.name,
                    "line": node.lineno,
                })

            elif isinstance(node, ast.AsyncFunctionDef):

                symbols.append({
                    "type": "async_function",
                    "name": node.name,
                    "line": node.lineno,
                })

            elif isinstance(node, (ast.Import, ast.ImportFrom)):

                symbols.append({
                    "type": "import",
                    "name": self._get_import_name(node),
                    "line": node.lineno,
                })

        return symbols

    @staticmethod
    def _get_import_name(node) -> str:

        if isinstance(node, ast.Import):

            return ", ".join(
                alias.name
                for alias in node.names
            )

        if isinstance(node, ast.ImportFrom):

            module = node.module or ""

            names = ", ".join(
                alias.name
                for alias in node.names
            )

            return f"{module}: {names}"

        return ""