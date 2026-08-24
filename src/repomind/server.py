from pathlib import Path

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("RepoMind")


# For now, inspect the current working directory.
# We will make this configurable later.
REPO_PATH = Path.cwd()


@mcp.tool()
def list_files() -> str:
    """
    List files and directories in the repository.

    Hidden Git files, virtual environments, and Python cache
    directories are excluded.
    """

    lines = []

    ignored_directories = {
        ".git",
        "venv",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
    }

    for path in sorted(REPO_PATH.rglob("*")):

        # Ignore directories and files inside ignored directories
        if any(part in ignored_directories for part in path.parts):
            continue

        relative_path = path.relative_to(REPO_PATH)

        if path.is_dir():
            lines.append(f"[DIR]  {relative_path}")
        else:
            lines.append(f"[FILE] {relative_path}")

    if not lines:
        return "Repository is empty."

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()