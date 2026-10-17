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

    def search_symbols(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[dict]:
        """
        Search for symbols across Python source files
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
            "build",
            "dist",
        }

        query_lower = query.lower()

        for path in sorted(
            self.repository_root.rglob("*.py")
        ):

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

                if (
                    query_lower in symbol["name"].lower()
                    or
                    query_lower in symbol[
                        "qualified_name"
                    ].lower()
                ):

                    results.append({
                        "file": str(relative_path),
                        **symbol,
                    })

                    if len(results) >= max_results:
                        return results

        return results

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

    def find_references(
        self,
        relative_path: str,
    ) -> list[dict]:
        """
        Extract meaningful symbol references from a Python file.
        """

        path = self._resolve_python_file(relative_path)

        source = path.read_text(
            encoding="utf-8"
        )

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(
                f"Could not parse {relative_path}: {exc}"
            )

        references = []

        for node in ast.walk(tree):

            # Function/class calls:
            #
            # get_repository()
            # Repository(...)
            #
            if isinstance(node, ast.Call):

                target = node.func

                if isinstance(target, ast.Name):

                    references.append({
                        "symbol_name": target.id,
                        "line": target.lineno,
                        "reference_type": "call",
                    })

                elif isinstance(target, ast.Attribute):

                    references.append({
                        "symbol_name": target.attr,
                        "line": target.lineno,
                        "reference_type": "method_call",
                    })

            # Imports:
            #
            # from repository import Repository
            #
            elif isinstance(node, ast.Import):

                for alias in node.names:

                    references.append({
                        "symbol_name": alias.name,
                        "line": node.lineno,
                        "reference_type": "import",
                    })

            elif isinstance(node, ast.ImportFrom):

                for alias in node.names:

                    references.append({
                        "symbol_name": alias.name,
                        "line": node.lineno,
                        "reference_type": "import",
                    })

            # Attribute access:
            #
            # repository.root
            #
            elif isinstance(node, ast.Attribute):

                # Avoid duplicating attributes that were already
                # recorded as method calls.
                if isinstance(node.ctx, ast.Load):

                    parent_is_call = False

                    for parent in ast.walk(tree):

                        if (
                            isinstance(parent, ast.Call)
                            and parent.func is node
                        ):
                            parent_is_call = True
                            break

                    if not parent_is_call:

                        references.append({
                            "symbol_name": node.attr,
                            "line": node.lineno,
                            "reference_type": "attribute",
                        })

            # Direct symbol references:
            #
            # Repository
            # analyzer
            #
            elif isinstance(node, ast.Name):

                # Skip names that are function calls.
                parent_is_call = False

                for parent in ast.walk(tree):

                    if (
                        isinstance(parent, ast.Call)
                        and parent.func is node
                    ):
                        parent_is_call = True
                        break

                if not parent_is_call:

                    references.append({
                        "symbol_name": node.id,
                        "line": node.lineno,
                        "reference_type": "reference",
                    })

        return references

    def find_imports(
        self,
        relative_path: str,
    ) -> list[dict]:
        """
        Extract Python imports from a file, preserving relative-import
        metadata so dependencies can be resolved in package context.
        """
        path = self._resolve_python_file(relative_path)

        source = path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(
                f"Could not parse {relative_path}: {exc}"
            ) from exc

        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "module": alias.name,
                        "line": node.lineno,
                        "type": "import",
                        "level": 0,
                    })

            elif isinstance(node, ast.ImportFrom):
                imports.append({
                    "module": node.module or "",
                    "line": node.lineno,
                    "type": "from_import",
                    "level": node.level,
                })

        return imports

    def resolve_import(
        self,
        module_name: str,
    ) -> str | None:
        """
        Resolve a Python module name to a file
        inside the configured repository.
        """

        if not module_name or not module_name.strip():
            return None
        parts = module_name.split(".")
        relative = Path(*parts)

        candidates = [
            # Standard repository layout
            self.repository_root / relative.with_suffix(".py"),

            # Standard package layout
            self.repository_root / relative / "__init__.py",

            # src/ layout
            self.repository_root
            / "src"
            / relative.with_suffix(".py"),

            # src/ package layout
            self.repository_root
            / "src"
            / relative
            / "__init__.py",
        ]

        for candidate in candidates:

            if candidate.exists():

                return str(
                    candidate.relative_to(
                        self.repository_root
                    )
                ).replace("\\", "/")

        return None

    def _get_relative_import_names(
        self,
        source_path: Path,
        line_number: int,
    ) -> list[str]:
        """Return imported names for a relative import statement."""
        source = source_path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(
                f"Could not parse {source_path}: {exc}"
            ) from exc

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.lineno == line_number
                and node.level > 0
            ):
                return [
                    alias.name
                    for alias in node.names
                    if alias.name != "*"
                ]

        return []

    def find_dependencies(
        self,
        relative_path: str,
    ) -> list[dict]:
        """
        Find local repository files imported by a Python file,
        including relative imports.
        """
        imports = self.find_imports(relative_path)
        source_path = self._resolve_python_file(relative_path)

        dependencies = []

        for item in imports:
            module_name = item["module"]
            level = item.get("level", 0)

            if level > 0:
                # Determine the package containing the source file.
                source_dir = source_path.parent
                package_parts = []

                while (
                    source_dir != self.repository_root
                    and (source_dir / "__init__.py").is_file()
                ):
                    package_parts.insert(0, source_dir.name)
                    source_dir = source_dir.parent

                # Relative imports require a package context.
                if not package_parts:
                    continue

                # One dot refers to the current package.
                # Two dots refer to its parent, and so on.
                ascend = level - 1

                if ascend >= len(package_parts):
                    continue

                base_parts = package_parts[:len(package_parts) - ascend]

                if module_name:
                    base_parts.extend(module_name.split("."))

                # For "from . import helper", resolve each imported
                # name against the current package rather than its
                # __init__.py file.
                if not module_name:
                    imported_names = self._get_relative_import_names(
                        source_path,
                        item["line"],
                    )

                    for imported_name in imported_names:
                        candidate_module = ".".join(
                            [*base_parts, imported_name]
                        )
                        candidate_file = self.resolve_import(
                            candidate_module
                        )

                        if candidate_file:
                            dependencies.append({
                                "target_file": candidate_file,
                                "line": item["line"],
                                "type": item["type"],
                            })

                    continue

                resolved_module = ".".join(base_parts)

                target_file = self.resolve_import(resolved_module)

                if target_file:
                    dependencies.append({
                        "target_file": target_file,
                        "line": item["line"],
                        "type": item["type"],
                    })

                continue

            # Absolute imports: resolve the module name directly.
            target_file = self.resolve_import(module_name)

            if target_file is None:
                continue

            dependencies.append({
                "target_file": target_file,
                "line": item["line"],
                "type": item["type"],
            })

        return dependencies