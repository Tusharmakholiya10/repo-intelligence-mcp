
from pathlib import Path

import pytest

from repomind.analyzer import PythonAnalyzer


def test_find_dependencies_detects_regular_import(
    tmp_path: Path,
):
    (tmp_path / "main.py").write_text(
        "import helper\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text(
        "def help_function():\n    return 1\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    dependencies = analyzer.find_dependencies("main.py")

    assert dependencies == [
        {
            "target_file": "helper.py",
            "line": 1,
            "type": "import",
        }
    ]


def test_find_dependencies_detects_from_import(
    tmp_path: Path,
):
    (tmp_path / "main.py").write_text(
        "from helper import help_function\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text(
        "def help_function():\n    return 1\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    dependencies = analyzer.find_dependencies("main.py")

    assert dependencies == [
        {
            "target_file": "helper.py",
            "line": 1,
            "type": "from_import",
        }
    ]


def test_find_dependencies_reflects_changed_import(
    tmp_path: Path,
):
    source_file = tmp_path / "main.py"

    source_file.write_text(
        "import helper\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text(
        "value = 2\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    initial_dependencies = analyzer.find_dependencies(
        "main.py"
    )

    assert [
        dependency["target_file"]
        for dependency in initial_dependencies
    ] == ["helper.py"]

    source_file.write_text(
        "import utils\n",
        encoding="utf-8",
    )

    updated_dependencies = analyzer.find_dependencies(
        "main.py"
    )

    assert [
        dependency["target_file"]
        for dependency in updated_dependencies
    ] == ["utils.py"]


def test_find_dependencies_ignores_unresolved_import(
    tmp_path: Path,
):
    (tmp_path / "main.py").write_text(
        "import helper\nimport third_party_package\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    dependencies = analyzer.find_dependencies("main.py")

    assert [
        dependency["target_file"]
        for dependency in dependencies
    ] == ["helper.py"]


def test_find_dependencies_rejects_invalid_python(
    tmp_path: Path,
):
    (tmp_path / "main.py").write_text(
        "def broken(:\n    pass\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    with pytest.raises(
        ValueError,
        match="Could not parse main.py",
    ):
        analyzer.find_dependencies("main.py")
