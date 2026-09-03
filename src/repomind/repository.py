from pathlib import Path


class Repository:
    """Represents the repository analyzed by RepoMind."""

    IGNORED_DIRECTORIES = {
        ".git",
        ".repomind",
        "venv",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
    }

    IGNORED_FILE_NAMES = {
        ".env",
    }

    IGNORED_FILE_SUFFIXES = {
        ".egg-info",
    }

    SEARCHABLE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".cs",
        ".sql",
        ".md",
        ".json",
        ".yaml",
        ".yml",
    }

    MAX_FILE_SIZE = 1_000_000  # 1 MB

    def __init__(self, path: str):
        """Initialize the repository."""

        self.root = Path(path).resolve()

        if not self.root.exists():
            raise ValueError(
                f"Repository does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise ValueError(
                f"Repository path is not a directory: {self.root}"
            )

    def _is_ignored(self, path: Path) -> bool:
        """Check whether a path belongs to an ignored directory or file."""

        # Ignore directories anywhere in the path.
        if any(
            part in self.IGNORED_DIRECTORIES
            for part in path.parts
        ):
            return True

        # Ignore explicitly sensitive file names.
        if path.name in self.IGNORED_FILE_NAMES:
            return True

        # Ignore generated/package metadata directories,
        # and anything nested inside them
        # (e.g. example.egg-info/PKG-INFO).
        if any(
            part.endswith(tuple(self.IGNORED_FILE_SUFFIXES))
            for part in path.parts
        ):
            return True

        # Ignore environment-specific files such as:
        # .env.local, .env.production, .env.test
        if path.name.startswith(".env."):
            return True

        return False

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """
        Resolve a repository-relative path safely.

        Prevents path traversal outside the repository.
        """

        requested_path = Path(relative_path)

        if requested_path.is_absolute():
            raise ValueError(
                "Absolute paths are not allowed."
            )

        resolved_path = (
            self.root / requested_path
        ).resolve()

        try:
            resolved_path.relative_to(self.root)
        except ValueError:
            raise ValueError(
                "Access denied: path is outside the repository."
            )

        return resolved_path

    def list_files(self) -> list[str]:
        """List files and directories in the repository."""

        results = []

        for path in sorted(self.root.rglob("*")):

            if self._is_ignored(path):
                continue

            relative_path = path.relative_to(
                self.root
            )

            if path.is_dir():
                results.append(
                    f"[DIR]  {relative_path}"
                )
            else:
                results.append(
                    f"[FILE] {relative_path}"
                )

        return results

    def read_file(self, relative_path: str) -> str:
        """Safely read a UTF-8 text file."""

        path = self._resolve_safe_path(
            relative_path
        )

        if not path.exists():
            raise ValueError(
                f"File not found: {relative_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {relative_path}"
            )

        if self._is_ignored(path):
            raise ValueError(
                "Access denied: file is inside "
                "an ignored directory."
            )

        file_size = path.stat().st_size

        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File is too large. Maximum allowed "
                f"size is {self.MAX_FILE_SIZE} bytes."
            )

        try:
            return path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            raise ValueError(
                "File is not a valid UTF-8 text file."
            )

    def search_code(
        self,
        query: str,
        max_results: int = 50,
    ) -> list[str]:
        """
        Search for a text query across source files.

        Returns matching file paths, line numbers,
        and source lines.
        """

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if max_results < 1:
            raise ValueError(
                "max_results must be at least 1."
            )

        results = []

        query_lower = query.lower()

        for path in sorted(
            self.root.rglob("*")
        ):

            if not path.is_file():
                continue

            if self._is_ignored(path):
                continue

            if (
                path.suffix.lower()
                not in self.SEARCHABLE_EXTENSIONS
            ):
                continue

            try:
                content = path.read_text(
                    encoding="utf-8"
                )
            except (
                UnicodeDecodeError,
                OSError,
            ):
                continue

            for line_number, line in enumerate(
                content.splitlines(),
                start=1,
            ):

                if query_lower in line.lower():

                    relative_path = (
                        path.relative_to(
                            self.root
                        )
                    )

                    results.append(
                        f"{relative_path}:"
                        f"{line_number}: "
                        f"{line.strip()}"
                    )

                    if (
                        len(results)
                        >= max_results
                    ):
                        return results

        return results