import subprocess
from pathlib import Path


class GitManager:
    """Provide Git history and repository information."""

    def __init__(self, repository_root: Path):
        self.repository_root = repository_root.resolve()

    def _run_git(self, *args: str) -> str:
        """Run a Git command inside the repository."""

        result = subprocess.run(
            ["git", *args],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    def get_history(
        self,
        max_results: int = 10,
    ) -> str:
        """Return recent Git commits."""

        return self._run_git(
            "log",
            f"-{max_results}",
            "--date=short",
            "--pretty=format:%h | %ad | %an | %s",
        )

    def get_commit(
        self,
        commit_hash: str,
    ) -> str:
        """Return details about a Git commit."""

        return self._run_git(
            "show",
            "--stat",
            "--oneline",
            commit_hash,
        )

    def get_file_history(
        self,
        relative_path: str,
        max_results: int = 10,
    ) -> str:
        """Return Git history for a specific file."""

        return self._run_git(
            "log",
            f"-{max_results}",
            "--date=short",
            "--pretty=format:%h | %ad | %an | %s",
            "--",
            relative_path,
        )

    def get_diff(
        self,
        commit_hash: str,
    ) -> str:
        """Return the diff introduced by a commit."""

        return self._run_git(
            "show",
            "--format=fuller",
            commit_hash,
        )