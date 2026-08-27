import os

from mcp.server.fastmcp import FastMCP

from repomind.repository import Repository
from repomind.analyzer import PythonAnalyzer
from repomind.indexer import CodeIndexer
from repomind.git import GitManager
from datetime import datetime

mcp = FastMCP("RepoMind")


def get_repository() -> Repository:
    """
    Create a Repository instance from the configured
    REPOMIND_REPO environment variable.
    """

    repo_path = os.getenv("REPOMIND_REPO", ".")

    return Repository(repo_path)

@mcp.tool()
def get_git_history(
    max_results: int = 10,
) -> str:
    """
    Return recent Git commits from the repository.
    """

    repository = get_repository()

    git = GitManager(repository.root)

    try:
        history = git.get_history(max_results)

    except Exception as error:
        return f"Git history error: {error}"

    if not history:
        return "No Git history found."

    return history

@mcp.tool()
def get_git_commit(
    commit_hash: str,
) -> str:
    """
    Return details and changed files for a Git commit.
    """

    repository = get_repository()

    git = GitManager(repository.root)

    try:
        return git.get_commit(commit_hash)

    except Exception as error:
        return f"Git commit error: {error}"

@mcp.tool()
def get_git_file_history(
    relative_path: str,
    max_results: int = 10,
) -> str:
    """
    Return Git history for a specific repository file.
    """

    repository = get_repository()

    git = GitManager(repository.root)

    try:
        history = git.get_file_history(
            relative_path,
            max_results,
        )

    except Exception as error:
        return f"Git file history error: {error}"

    if not history:
        return (
            f"No Git history found for: "
            f"{relative_path}"
        )

    return history

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
            f"{symbol['qualified_name']} "
            f"(lines "
            f"{symbol['line']}-{symbol['end_line']})"
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

@mcp.tool()
def search_symbols(
    query: str,
    max_results: int = 100,
) -> str:
    """
    Search the SQLite code index for classes,
    functions, methods and imports.
    """

    repository = get_repository()

    indexer = CodeIndexer(
        repository.root
    )

    results = indexer.search_symbols(
        query,
        max_results,
    )

    if not results:
        return (
            f"No indexed symbols found for: {query}\n"
            "Run index_repository() first."
        )

    lines = []

    for symbol in results:

        lines.append(
            f"{symbol['path']}: "
            f"{symbol['type']} "
            f"{symbol['qualified_name']} "
            f"(lines "
            f"{symbol['line']}-"
            f"{symbol['end_line']})"
        )

    return "\n".join(lines)

@mcp.tool()
def index_repository() -> str:
    """
    Build or update the SQLite code index
    for the configured repository.
    """

    repository = get_repository()

    analyzer = PythonAnalyzer(
        repository.root
    )

    indexer = CodeIndexer(
        repository.root
    )

    indexed_files = 0
    indexed_symbols = 0

    python_files = []

    # --------------------------------------------------
    # PASS 1: Index all Python files and their symbols
    # --------------------------------------------------

    for path in repository.root.rglob("*.py"):

        if repository._is_ignored(path):
            continue

        relative_path = path.relative_to(
            repository.root
        ).as_posix()

        try:
            symbols = analyzer.analyze_file(
                str(relative_path)
            )
        except ValueError:
            continue

        stat = path.stat()

        modified_time = datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat()

        indexer.index_file(
            relative_path=str(relative_path),
            language="python",
            size=stat.st_size,
            modified_time=modified_time,
            symbols=symbols,
        )

        python_files.append(relative_path)

        indexed_files += 1
        indexed_symbols += len(symbols)

    # --------------------------------------------------
    # PASS 2: Index references and dependencies
    # --------------------------------------------------

    for relative_path in python_files:

        try:
            references = analyzer.find_references(
                str(relative_path)
            )

            dependencies = analyzer.find_dependencies(
                str(relative_path)
            )

        except ValueError:
            continue

        indexer.index_references(
            relative_path=str(relative_path),
            references=references,
        )

        indexer.index_dependencies(
            relative_path=str(relative_path),
            dependencies=dependencies,
        )

    return (
        f"Repository indexed successfully.\n"
        f"Files indexed: {indexed_files}\n"
        f"Symbols indexed: {indexed_symbols}\n"
        f"Database: {indexer.database_path}"
    )

@mcp.tool()
def index_stats() -> str:
    """
    Return statistics about the RepoMind code index.
    """

    repository = get_repository()

    indexer = CodeIndexer(
        repository.root
    )

    stats = indexer.get_stats()

    return (
        f"Indexed files: {stats['files']}\n"
        f"Indexed symbols: {stats['symbols']}\n"
        f"Indexed references: {stats['references']}\n"
        f"Database: {stats['database']}"
    )

@mcp.tool()
def find_usages(
    symbol_name: str,
    max_results: int = 100,
) -> str:
    """
    Find references to a symbol using the
    indexed repository.
    """

    repository = get_repository()

    indexer = CodeIndexer(
        repository.root
    )

    usages = indexer.find_usages(
        symbol_name,
        max_results,
    )

    if not usages:
        return (
            f"No usages found for: {symbol_name}\n"
            "Run index_repository() first."
        )

    lines = []

    for usage in usages:

        lines.append(
            f"{usage['path']}:"
            f"{usage['line']} "
            f"[{usage['reference_type']}] "
            f"{usage['symbol_name']}"
        )

    return "\n".join(lines)

@mcp.tool()
def get_dependencies(
    relative_path: str,
    max_results: int = 100,
) -> str:
    """
    Return local repository files that a source
    file depends on.
    """

    repository = get_repository()

    indexer = CodeIndexer(
        repository.root
    )

    dependencies = indexer.get_dependencies(
        relative_path,
        max_results,
    )

    if not dependencies:
        return (
            f"No local dependencies found for: "
            f"{relative_path}"
        )

    lines = []

    for dependency in dependencies:

        lines.append(
            f"{relative_path} -> "
            f"{dependency['path']} "
            f"[{dependency['dependency_type']}] "
            f"line {dependency['line']}"
        )

    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run()