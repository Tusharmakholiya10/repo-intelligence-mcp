from pathlib import Path

import pytest

from repomind.repository import Repository


@pytest.fixture
def test_repository(tmp_path: Path) -> Repository:
    """Create a temporary repository for testing."""

    (tmp_path / "src").mkdir()

    (tmp_path / "src" / "main.py").write_text(
        """
def hello():
    return "hello"

def goodbye():
    return "goodbye"
""",
        encoding="utf-8",
    )

    (tmp_path / "README.md").write_text(
        "# Test Repository\n",
        encoding="utf-8",
    )

    return Repository(tmp_path)


def test_list_files(test_repository):
    """Repository should list project files."""

    files = test_repository.list_files()

    assert "[FILE] README.md" in files
    assert "[DIR]  src" in files
    assert any(
        file.endswith("src\\main.py")
        or file.endswith("src/main.py")
        for file in files
    )


def test_read_file(test_repository):
    """Repository should read a file."""

    content = test_repository.read_file("README.md")

    assert "# Test Repository" in content


def test_search_code(test_repository):
    """Repository should find text across source files."""

    results = test_repository.search_code("hello")

    assert any("main.py" in result for result in results)


def test_search_code_no_match(test_repository):
    """Repository should return no results for missing text."""

    results = test_repository.search_code(
        "this_text_does_not_exist"
    )

    assert results == []


def test_read_file_missing(test_repository):
    """Reading a missing file should raise an error."""

    with pytest.raises(Exception):
        test_repository.read_file("missing.py")


def test_ignored_directories(test_repository):
    """Ignored directories should not appear in repository listings."""

    ignored_dir = test_repository.root / ".git"
    ignored_dir.mkdir()

    ignored_file = ignored_dir / "config"
    ignored_file.write_text(
        "secret",
        encoding="utf-8",
    )

    files = test_repository.list_files()

    assert not any(".git" in file for file in files)


def test_sensitive_env_files_are_hidden(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    (repo_path / ".env").write_text(
        "GEMINI_API_KEY=secret",
        encoding="utf-8",
    )

    (repo_path / ".env.test").write_text(
        "fake-secret=test",
        encoding="utf-8",
    )

    (repo_path / ".env.production").write_text(
        "PRODUCTION_SECRET=secret",
        encoding="utf-8",
    )

    (repo_path / "README.md").write_text(
        "# Test repository",
        encoding="utf-8",
    )

    repository = Repository(str(repo_path))

    files = repository.list_files()

    assert ".env" not in "\n".join(files)
    assert ".env.test" not in "\n".join(files)
    assert ".env.production" not in "\n".join(files)


def test_sensitive_env_files_cannot_be_read(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    (repo_path / ".env").write_text(
        "GEMINI_API_KEY=secret",
        encoding="utf-8",
    )

    (repo_path / ".env.test").write_text(
        "fake-secret=test",
        encoding="utf-8",
    )

    repository = Repository(str(repo_path))

    try:
        repository.read_file(".env")
        assert False, "Expected .env access to be blocked"
    except ValueError as exc:
        assert "Access denied" in str(exc)

    try:
        repository.read_file(".env.test")
        assert False, "Expected .env.test access to be blocked"
    except ValueError as exc:
        assert "Access denied" in str(exc)


def test_egg_info_directory_is_ignored(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    egg_info = repo_path / "example.egg-info"
    egg_info.mkdir()

    (egg_info / "PKG-INFO").write_text(
        "generated metadata",
        encoding="utf-8",
    )

    repository = Repository(str(repo_path))

    files = repository.list_files()

    assert "example.egg-info" not in "\n".join(files)
    assert "PKG-INFO" not in "\n".join(files)


def test_path_traversal_is_blocked(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    outside_file = tmp_path / "outside.txt"

    outside_file.write_text(
        "secret outside repository",
        encoding="utf-8",
    )

    repository = Repository(str(repo_path))

    try:
        repository.read_file("../outside.txt")
        assert False, "Expected path traversal to be blocked"
    except ValueError as exc:
        assert "outside the repository" in str(exc)


def test_normal_file_remains_readable(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    source_file = repo_path / "example.py"

    source_file.write_text(
        "print('hello')",
        encoding="utf-8",
    )

    repository = Repository(str(repo_path))

    assert repository.read_file("example.py") == "print('hello')"