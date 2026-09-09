import os
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from repomind.repository import Repository
from repomind.analyzer import PythonAnalyzer
from repomind.indexer import CodeIndexer
from repomind.git import GitManager
from repomind.chunker import CodeChunker
from repomind.embeddings import EmbeddingEngine


mcp = FastMCP("RepoMind")


def get_repository() -> Repository:
    """
    Create a Repository instance from the configured
    REPOMIND_REPO environment variable.
    """

    repo_path = os.getenv(
        "REPOMIND_REPO",
        ".",
    )

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
        history = git.get_history(
            max_results
        )

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
        return git.get_commit(
            commit_hash
        )

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
def analyze_python_file(
    path: str,
) -> str:
    """
    Analyze a Python file and return its classes,
    functions, async functions and imports.
    """

    repository = get_repository()

    analyzer = PythonAnalyzer(
        repository.root
    )

    symbols = analyzer.analyze_file(path)

    if not symbols:
        return (
            f"No symbols found in {path}"
        )

    lines = []

    for symbol in symbols:

        lines.append(
            f"{symbol['type']}: "
            f"{symbol['qualified_name']} "
            f"(lines "
            f"{symbol['line']}-"
            f"{symbol['end_line']})"
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
def read_file(
    path: str,
) -> str:
    """
    Read a text file from the configured repository.

    The path must be relative to the repository root.
    """

    repository = get_repository()

    return repository.read_file(path)


@mcp.tool()
def search_code(
    query: str,
    max_results: int = 50,
) -> str:
    """
    Search for a text query across source files
    in the configured repository.
    """

    repository = get_repository()

    results = repository.search_code(
        query,
        max_results,
    )

    if not results:
        return (
            f"No matches found for: {query}"
        )

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
    Build or update the SQLite code index for the configured repository.

    Only new or modified files are analyzed.

    Semantic chunks are generated and embedded only for
    new, modified, or not-yet-semantically-indexed files.
    """

    repository = get_repository()

    analyzer = PythonAnalyzer(
        repository.root
    )

    indexer = CodeIndexer(
        repository.root
    )

    chunker = CodeChunker()

    try:
        embedding_engine = EmbeddingEngine()

    except ValueError as error:
        return (
            f"Semantic indexing unavailable: {error}"
        )

    indexed_files = 0
    indexed_symbols = 0
    semantic_chunks_indexed = 0
    skipped_files = 0
    deleted_files = 0
    embedding_errors = 0

    current_files = set()

    try:

        for path in repository.root.rglob("*.py"):

            if repository._is_ignored(path):
                continue

            relative_path = path.relative_to(
                repository.root
            )

            relative_path_str = str(
                relative_path
            ).replace(
                "\\",
                "/",
            )

            current_files.add(
                relative_path_str
            )

            stat = path.stat()

            modified_time = (
                datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat()
            )

            # -------------------------------------------------
            # Determine whether this file needs work.
            # -------------------------------------------------

            needs_code_reindex = (
                indexer.needs_reindex(
                    relative_path_str,
                    stat.st_size,
                    modified_time,
                )
            )

            needs_semantic_reindex = (
                indexer.needs_semantic_reindex(
                    relative_path_str
                )
            )

            if (
                not needs_code_reindex
                and not needs_semantic_reindex
            ):
                skipped_files += 1
                continue

            try:

                # -------------------------------------------------
                # Analyze source
                # -------------------------------------------------

                symbols = analyzer.analyze_file(
                    relative_path_str
                )

                references = (
                    analyzer.find_references(
                        relative_path_str
                    )
                )

                dependencies = (
                    analyzer.find_dependencies(
                        relative_path_str
                    )
                )

                # -------------------------------------------------
                # Read source
                # -------------------------------------------------

                source_text = path.read_text(
                    encoding="utf-8"
                )

                # -------------------------------------------------
                # Create semantic chunks
                # -------------------------------------------------

                chunks = (
                    chunker.chunk_python_file(
                        relative_path_str,
                        source_text,
                        symbols,
                    )
                )

                # -------------------------------------------------
                # Generate embeddings
                # -------------------------------------------------

                chunk_texts = [
                    chunk.content
                    for chunk in chunks
                ]

                embeddings = (
                    embedding_engine.embed_documents(
                        chunk_texts
                    )
                    if chunk_texts
                    else []
                )

                if len(embeddings) != len(
                    chunks
                ):
                    raise RuntimeError(
                        "Embedding count does not match "
                        "semantic chunk count."
                    )

                semantic_chunk_records = []

                for chunk, embedding in zip(
                    chunks,
                    embeddings,
                ):

                    semantic_chunk_records.append(
                        {
                            "start_line": (
                                chunk.start_line
                            ),
                            "end_line": (
                                chunk.end_line
                            ),
                            "symbol_name": (
                                chunk.symbol_name
                            ),
                            "symbol_type": (
                                chunk.symbol_type
                            ),
                            "content": chunk.content,
                            "embedding": embedding,
                        }
                    )

            except (
                ValueError,
                UnicodeDecodeError,
                OSError,
                RuntimeError,
            ):
                embedding_errors += 1
                continue

            # -------------------------------------------------
            # Persist normal index.
            # -------------------------------------------------

            indexer.index_file(
                relative_path=relative_path_str,
                language="python",
                size=stat.st_size,
                modified_time=modified_time,
                symbols=symbols,
            )

            indexer.index_references(
                relative_path=relative_path_str,
                references=references,
            )

            indexer.index_dependencies(
                relative_path=relative_path_str,
                dependencies=dependencies,
            )

            # -------------------------------------------------
            # Persist semantic index.
            # -------------------------------------------------

            indexer.index_semantic_chunks(
                relative_path=relative_path_str,
                chunks=semantic_chunk_records,
            )

            indexer.mark_semantic_indexed(
                relative_path_str
            )

            indexed_files += 1
            indexed_symbols += len(symbols)
            semantic_chunks_indexed += len(
                semantic_chunk_records
            )

        # ---------------------------------------------------------
        # Remove deleted files.
        # Cascading deletes remove their semantic chunks too.
        # ---------------------------------------------------------

        with indexer._connect() as connection:

            rows = connection.execute(
                "SELECT path FROM files"
            ).fetchall()

            indexed_paths = {
                str(
                    row["path"]
                ).replace(
                    "\\",
                    "/",
                )
                for row in rows
            }

            deleted_paths = (
                indexed_paths - current_files
            )

            for deleted_path in deleted_paths:

                connection.execute(
                    """
                    DELETE FROM files
                    WHERE path = ?
                    """,
                    (deleted_path,),
                )

            deleted_files = len(
                deleted_paths
            )

            connection.commit()

    finally:
        embedding_engine.close()

    stats = indexer.get_stats()

    return (
        f"Repository indexed successfully.\n"
        f"Files updated: {indexed_files}\n"
        f"Files skipped: {skipped_files}\n"
        f"Files deleted: {deleted_files}\n"
        f"Symbols indexed: {indexed_symbols}\n"
        f"Semantic chunks indexed: "
        f"{semantic_chunks_indexed}\n"
        f"Semantic indexing errors: "
        f"{embedding_errors}\n"
        f"Semantic indexed files: "
        f"{stats['semantic_indexed_files']}\n"
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
        f"Indexed dependencies: "
        f"{stats.get('dependencies', 0)}\n"
        f"Semantic chunks: "
        f"{stats.get('semantic_chunks', 0)}\n"
        f"Semantic indexed files: "
        f"{stats.get('semantic_indexed_files', 0)}\n"
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

    normalized_source = str(
        relative_path
    ).replace(
        "\\",
        "/",
    )

    for dependency in dependencies:

        lines.append(
            f"{normalized_source} -> "
            f"{dependency['path']} "
            f"[{dependency['dependency_type']}] "
            f"line {dependency['line']}"
        )

    return "\n".join(lines)

@mcp.tool()
def semantic_search(
    query: str,
    max_results: int = 5,
    min_similarity: float = 0.0,
) -> str:
    """
    Search repository code semantically using Gemini embeddings.

    Returns the most semantically relevant code chunks.
    """

    if not query or not query.strip():
        return "Semantic search query cannot be empty."

    if max_results < 1:
        return "max_results must be at least 1."

    if not 0.0 <= min_similarity <= 1.0:
        return (
            "min_similarity must be between 0.0 and 1.0."
        )

    repository = get_repository()

    indexer = CodeIndexer(
        repository.root
    )

    stats = indexer.get_stats()

    if stats.get("semantic_chunks", 0) == 0:
        return (
            "Semantic index is empty. "
            "Run index_repository() first."
        )

    try:
        embedding_engine = EmbeddingEngine()

        try:
            query_embedding = (
                embedding_engine.embed_query(
                    query
                )
            )

        finally:
            embedding_engine.close()

    except Exception as error:
        return (
            f"Semantic embedding error: {error}"
        )

    try:
        results = indexer.semantic_search(
            query_embedding=query_embedding,
            max_results=max_results,
            min_similarity=min_similarity,
            query_text=query,
        )

    except ValueError as error:
        return (
            f"Semantic search error: {error}"
        )

    if not results:
        return (
            f"No semantic matches found for: {query}"
        )

    lines = [
        f"Semantic search results for: {query}",
        ""
    ]

    for index, result in enumerate(
        results,
        start=1,
    ):

        similarity = result["similarity"]
        lexical_score = result.get(
            "lexical_score",
            0.0,
        )
        relevance_score = result.get(
            "score",
            similarity,
        )

        lines.append(
            f"{index}. "
            f"{result['path']}:"
            f"{result['start_line']}-"
            f"{result['end_line']}"
        )

        if result["symbol_name"]:
            lines.append(
                f"   Symbol: "
                f"{result['symbol_name']}"
            )

        if result["symbol_type"]:
            lines.append(
                f"   Type: "
                f"{result['symbol_type']}"
            )

        lines.append(
            f"   Semantic similarity: "
            f"{similarity:.4f}"
        )

        lines.append(
            f"   Lexical relevance: "
            f"{lexical_score:.4f}"
        )

        lines.append(
            f"   Combined score: "
            f"{relevance_score:.4f}"
        )

        lines.append(
            "   Code:"
        )

        lines.append(
            "\n".join(
                f"   {line}"
                for line in result["content"].splitlines()
            )
        )

        lines.append("")

    return "\n".join(lines)

def main():
    """Start the RepoMind MCP server."""

    mcp.run()


if __name__ == "__main__":
    main()