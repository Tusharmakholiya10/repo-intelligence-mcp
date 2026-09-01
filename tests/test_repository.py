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


def test_path_traversal_is_blocked(test_repository):
    """Repository should prevent access outside the repository."""

    with pytest.raises(Exception):
        test_repository.read_file("../outside.txt")


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