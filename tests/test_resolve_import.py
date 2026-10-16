from pathlib import Path

from repomind.analyzer import PythonAnalyzer


def test_resolve_import_finds_root_module(tmp_path: Path):
    (tmp_path / "helper.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    assert analyzer.resolve_import("helper") == "helper.py"


def test_resolve_import_finds_root_package(tmp_path: Path):
    package = tmp_path / "mypackage"
    package.mkdir()
    (package / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    assert analyzer.resolve_import("mypackage") == (
        "mypackage/__init__.py"
    )


def test_resolve_import_finds_src_module(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "helper.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    assert analyzer.resolve_import("helper") == "src/helper.py"


def test_resolve_import_finds_src_package(tmp_path: Path):
    package = tmp_path / "src" / "mypackage"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    assert analyzer.resolve_import("mypackage") == (
        "src/mypackage/__init__.py"
    )


def test_resolve_import_finds_nested_module(tmp_path: Path):
    package = tmp_path / "mypackage"
    package.mkdir()
    (package / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (package / "helper.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    assert analyzer.resolve_import("mypackage.helper") == (
        "mypackage/helper.py"
    )


def test_resolve_import_returns_none_for_missing_module(
    tmp_path: Path,
):
    analyzer = PythonAnalyzer(tmp_path)

    assert analyzer.resolve_import("third_party_package") is None