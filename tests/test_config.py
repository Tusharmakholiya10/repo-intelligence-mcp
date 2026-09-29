from __future__ import annotations

from pathlib import Path

import pytest

from repomind.config import (
    ConfigError,
    RepoMindConfig,
)


def test_default_configuration(tmp_path: Path):
    config = RepoMindConfig.from_env(
        {
            "REPOMIND_REPO": str(tmp_path),
        }
    )

    assert config.repo_path == tmp_path.resolve()
    assert config.gemini_api_key is None
    assert (
        config.gemini_model
        == RepoMindConfig.DEFAULT_GEMINI_MODEL
    )
    assert (
        config.max_file_size
        == RepoMindConfig.DEFAULT_MAX_FILE_SIZE
    )


def test_repository_path_is_required_to_exist(
    tmp_path: Path,
):
    missing = tmp_path / "missing-repository"

    with pytest.raises(
        ConfigError,
        match="does not exist",
    ):
        RepoMindConfig.from_env(
            {
                "REPOMIND_REPO": str(missing),
            }
        )


def test_repository_path_must_be_directory(
    tmp_path: Path,
):
    file_path = tmp_path / "repository.txt"
    file_path.write_text(
        "test",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigError,
        match="is not a directory",
    ):
        RepoMindConfig.from_env(
            {
                "REPOMIND_REPO": str(file_path),
            }
        )


def test_empty_repository_path_is_rejected(
    tmp_path: Path,
):
    with pytest.raises(
        ConfigError,
        match="REPOMIND_REPO cannot be empty",
    ):
        RepoMindConfig.from_env(
            {
                "REPOMIND_REPO": "   ",
            }
        )


def test_gemini_api_key_is_trimmed(
    tmp_path: Path,
):
    config = RepoMindConfig.from_env(
        {
            "REPOMIND_REPO": str(tmp_path),
            "GEMINI_API_KEY": "  test-key  ",
        }
    )

    assert config.gemini_api_key == "test-key"


def test_empty_gemini_api_key_becomes_unset(
    tmp_path: Path,
):
    config = RepoMindConfig.from_env(
        {
            "REPOMIND_REPO": str(tmp_path),
            "GEMINI_API_KEY": "   ",
        }
    )

    assert config.gemini_api_key is None


def test_missing_gemini_key_is_rejected_when_required(
    tmp_path: Path,
):
    config = RepoMindConfig.from_env(
        {
            "REPOMIND_REPO": str(tmp_path),
        }
    )

    with pytest.raises(
        ConfigError,
        match="GEMINI_API_KEY is required",
    ):
        config.require_gemini_api_key()


def test_gemini_key_is_returned_when_configured(
    tmp_path: Path,
):
    config = RepoMindConfig.from_env(
        {
            "REPOMIND_REPO": str(tmp_path),
            "GEMINI_API_KEY": "test-key",
        }
    )

    assert (
        config.require_gemini_api_key()
        == "test-key"
    )


def test_empty_gemini_model_is_rejected(
    tmp_path: Path,
):
    with pytest.raises(
        ConfigError,
        match="GEMINI_MODEL cannot be empty",
    ):
        RepoMindConfig.from_env(
            {
                "REPOMIND_REPO": str(tmp_path),
                "GEMINI_MODEL": "   ",
            }
        )


def test_custom_gemini_model(
    tmp_path: Path,
):
    config = RepoMindConfig.from_env(
        {
            "REPOMIND_REPO": str(tmp_path),
            "GEMINI_MODEL": "custom-model",
        }
    )

    assert config.gemini_model == "custom-model"


def test_custom_max_file_size(
    tmp_path: Path,
):
    config = RepoMindConfig.from_env(
        {
            "REPOMIND_REPO": str(tmp_path),
            "REPOMIND_MAX_FILE_SIZE": "5000000",
        }
    )

    assert config.max_file_size == 5_000_000


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "1.5",
        "",
        "-10",
        "0",
    ],
)
def test_invalid_max_file_size_values(
    tmp_path: Path,
    value: str,
):
    with pytest.raises(
        ConfigError,
    ):
        RepoMindConfig.from_env(
            {
                "REPOMIND_REPO": str(tmp_path),
                "REPOMIND_MAX_FILE_SIZE": value,
            }
        )


def test_max_file_size_upper_bound(
    tmp_path: Path,
):
    too_large = (
        RepoMindConfig.MAX_FILE_SIZE_LIMIT
        + 1
    )

    with pytest.raises(
        ConfigError,
        match="must be between",
    ):
        RepoMindConfig.from_env(
            {
                "REPOMIND_REPO": str(tmp_path),
                "REPOMIND_MAX_FILE_SIZE": str(
                    too_large
                ),
            }
        )