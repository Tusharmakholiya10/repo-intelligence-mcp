import os

from mcp.server.fastmcp import FastMCP

from repomind.repository import Repository
from repomind.analyzer import PythonAnalyzer

mcp = FastMCP("RepoMind")


def get_repository() -> Repository:
    """
    Create a Repository instance from the configured
    REPOMIND_REPO environment variable.
    """

    repo_path = os.getenv("REPOMIND_REPO", ".")

    return Repository(repo_path)


@mcp.tool()
def analyze_python_file(path: str) -> str:
    """
    Analyze a Python file and return its classes,
    functions, async functions and imports.
    """

    repository = get_repository()

    analyzer = PythonAnalyzer(repository.root)

    symbols = analyzer.analyze_file(path)

    if not symbols:
        return f"No symbols found in {path}"

    lines = []

    for symbol in symbols:

        lines.append(
            f"{symbol['type']}: "
            f"{symbol['name']} "
            f"(line {symbol['line']})"
        )

    return "\n".join(lines)

@mcp.tool()
def list_files() -> str:
    """
    List files and directories in the configured repository.

    Hidden Git files, virtual environments, caches,
    and build directories are excluded.
    """

    repository = get_repository()

    files = repository.list_files()

    if not files:
        return "Repository is empty."

    return "\n".join(files)


@mcp.tool()
def read_file(path: str) -> str:
    """
    Read a text file from the configured repository.

    The path must be relative to the repository root.
    """

    repository = get_repository()

    return repository.read_file(path)


@mcp.tool()
def search_code(
    query: str,
    max_results: int = 50
) -> str:
    """
    Search for a text query across source files
    in the configured repository.
    """

    repository = get_repository()

    results = repository.search_code(
        query,
        max_results
    )

    if not results:
        return f"No matches found for: {query}"

    return "\n".join(results)


if __name__ == "__main__":
    mcp.run()