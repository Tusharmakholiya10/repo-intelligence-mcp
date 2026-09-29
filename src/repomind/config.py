from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """Raised when RepoMind runtime configuration is invalid."""


@dataclass(frozen=True)
class RepoMindConfig:
    """
    Validated RepoMind runtime configuration.

    Environment variables:

    REPOMIND_REPO
        Repository path. Defaults to the current directory.

    GEMINI_API_KEY
        Gemini API key. Optional during configuration loading,
        but required for semantic indexing/search.

    GEMINI_MODEL
        Gemini model used by the interactive agent.
        Defaults to gemini-3.8-flash.

    REPOMIND_MAX_FILE_SIZE
        Maximum readable file size in bytes.
        Defaults to 1,000,000 (1 MB).
    """

    repo_path: Path
    gemini_api_key: str | None
    gemini_model: str
    max_file_size: int

    DEFAULT_REPO_PATH = "."
    DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
    DEFAULT_MAX_FILE_SIZE = 1_000_000

    MIN_FILE_SIZE = 1
    MAX_FILE_SIZE_LIMIT = 50_000_000

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "RepoMindConfig":
        """
        Build and validate configuration from environment variables.

        A mapping can be supplied by tests; otherwise os.environ
        is used.
        """
        source = env if env is not None else os.environ

        repo_value = source.get(
            "REPOMIND_REPO",
            cls.DEFAULT_REPO_PATH,
        ).strip()

        if not repo_value:
            raise ConfigError(
                "REPOMIND_REPO cannot be empty."
            )

        repo_path = Path(
            repo_value
        ).expanduser().resolve()

        if not repo_path.exists():
            raise ConfigError(
                "REPOMIND_REPO does not exist: "
                f"{repo_path}"
            )

        if not repo_path.is_dir():
            raise ConfigError(
                "REPOMIND_REPO is not a directory: "
                f"{repo_path}"
            )

        api_key = source.get(
            "GEMINI_API_KEY"
        )

        if api_key is not None:
            api_key = api_key.strip()

            if not api_key:
                api_key = None

        model = source.get(
            "GEMINI_MODEL",
            cls.DEFAULT_GEMINI_MODEL,
        ).strip()

        if not model:
            raise ConfigError(
                "GEMINI_MODEL cannot be empty."
            )

        max_file_size = cls._parse_positive_int(
            source.get(
                "REPOMIND_MAX_FILE_SIZE",
                str(cls.DEFAULT_MAX_FILE_SIZE),
            ),
            "REPOMIND_MAX_FILE_SIZE",
        )

        if (
            max_file_size < cls.MIN_FILE_SIZE
            or max_file_size > cls.MAX_FILE_SIZE_LIMIT
        ):
            raise ConfigError(
                "REPOMIND_MAX_FILE_SIZE must be between "
                f"{cls.MIN_FILE_SIZE} and "
                f"{cls.MAX_FILE_SIZE_LIMIT} bytes."
            )

        return cls(
            repo_path=repo_path,
            gemini_api_key=api_key,
            gemini_model=model,
            max_file_size=max_file_size,
        )

    @staticmethod
    def _parse_positive_int(
        value: str,
        variable_name: str,
    ) -> int:
        """
        Parse an integer-valued environment variable.
        """
        try:
            parsed = int(value)
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ConfigError(
                f"{variable_name} must be an integer."
            ) from error

        if parsed <= 0:
            raise ConfigError(
                f"{variable_name} must be greater than 0."
            )

        return parsed

    def require_gemini_api_key(self) -> str:
        """
        Return the Gemini API key or raise a clear configuration error.

        Gemini is optional for basic repository/index inspection,
        but semantic indexing/search requires it.
        """
        if not self.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is required for "
                "semantic indexing and semantic search."
            )

        return self.gemini_api_key