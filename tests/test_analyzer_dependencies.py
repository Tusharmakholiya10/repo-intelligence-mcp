
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



def test_find_imports_preserves_relative_import_level(
    tmp_path: Path,
):
    package = tmp_path / "mypackage"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from . import helper\n"
        "from .utils import calculate\n"
        "from ..common import settings\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    imports = analyzer.find_imports("mypackage/main.py")

    assert imports == [
        {
            "module": "",
            "line": 1,
            "type": "from_import",
            "level": 1,
        },
        {
            "module": "utils",
            "line": 2,
            "type": "from_import",
            "level": 1,
        },
        {
            "module": "common",
            "line": 3,
            "type": "from_import",
            "level": 2,
        },
    ]




def test_find_imports_records_relative_import_level(
    tmp_path: Path,
):
    package = tmp_path / "mypackage"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "import helper\n"
        "from helper import calculate\n"
        "from .utils import calculate\n"
        "from ..common import settings\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    imports = analyzer.find_imports("mypackage/main.py")

    assert imports == [
        {
            "module": "helper",
            "line": 1,
            "type": "import",
            "level": 0,
        },
        {
            "module": "helper",
            "line": 2,
            "type": "from_import",
            "level": 0,
        },
        {
            "module": "utils",
            "line": 3,
            "type": "from_import",
            "level": 1,
        },
        {
            "module": "common",
            "line": 4,
            "type": "from_import",
            "level": 2,
        },
    ]

def test_find_dependencies_resolves_same_package_import(
    tmp_path: Path,
):
    package = tmp_path / "mypackage"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(
        "from . import helper\n"
        "from .utils import calculate\n",
        encoding="utf-8",
    )
    (package / "helper.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )
    (package / "utils.py").write_text(
        "def calculate():\n    return 1\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    dependencies = analyzer.find_dependencies("mypackage/main.py")

    assert dependencies == [
        {
            "target_file": "mypackage/helper.py",
            "line": 1,
            "type": "from_import",
        },
        {
            "target_file": "mypackage/utils.py",
            "line": 2,
            "type": "from_import",
        },
    ]


def test_find_dependencies_resolves_parent_package_import(
    tmp_path: Path,
):
    package = tmp_path / "mypackage"
    subpackage = package / "subpackage"
    subpackage.mkdir(parents=True)

    (package / "__init__.py").write_text("", encoding="utf-8")
    (subpackage / "__init__.py").write_text("", encoding="utf-8")
    (subpackage / "main.py").write_text(
        "from .. import helper\n",
        encoding="utf-8",
    )
    (package / "helper.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    dependencies = analyzer.find_dependencies(
        "mypackage/subpackage/main.py"
    )

    assert dependencies == [
        {
            "target_file": "mypackage/helper.py",
            "line": 1,
            "type": "from_import",
        }
    ]

def test_find_dependencies_prefers_imported_module_in_package(
    tmp_path: Path,
):
    package = tmp_path / "mypackage"
    package.mkdir()

    (package / "__init__.py").write_text(
        "PACKAGE_INITIALIZED = True\n",
        encoding="utf-8",
    )
    (package / "main.py").write_text(
        "from mypackage import helper\n",
        encoding="utf-8",
    )
    (package / "helper.py").write_text(
        "HELPER_VALUE = 42\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    dependencies = analyzer.find_dependencies(
        "mypackage/main.py"
    )

    assert dependencies == [
        {
            "target_file": "mypackage/helper.py",
            "line": 1,
            "type": "from_import",
        }
    ]

def test_find_dependencies_preserves_duplicate_import_occurrences(
    tmp_path: Path,
):
    (tmp_path / "main.py").write_text(
        "import helper\n"
        "import helper\n"
        "from helper import calculate\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.py").write_text(
        "def calculate():\n"
        "    return 42\n",
        encoding="utf-8",
    )

    analyzer = PythonAnalyzer(tmp_path)

    dependencies = analyzer.find_dependencies("main.py")

    assert dependencies == [
        {
            "target_file": "helper.py",
            "line": 1,
            "type": "import",
        },
        {
            "target_file": "helper.py",
            "line": 2,
            "type": "import",
        },
        {
            "target_file": "helper.py",
            "line": 3,
            "type": "from_import",
        },
    ]
