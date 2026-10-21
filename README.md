# RepoMind — AI-Powered Codebase Intelligence MCP Server

RepoMind is a Python-based **Model Context Protocol (MCP) server** that helps AI coding agents understand, search, and navigate software repositories using structured code intelligence, Git history, dependency analysis, and hybrid semantic retrieval.

Instead of relying only on broad text searches, RepoMind combines source-code analysis, indexed symbols, dependency relationships, semantic search, and context packing to provide more relevant repository evidence to AI agents.

## Key Features

- **Repository exploration:** List files, read source code, and search code, comments, and documentation.
- **Python code analysis:** Extract classes, functions, methods, and imports using AST-based analysis.
- **Symbol intelligence:** Search indexed symbols and locate symbol usages.
- **Dependency analysis:** Identify local dependencies and explore relationships between files.
- **Git intelligence:** Inspect recent commits, individual commits, and file history.
- **Incremental indexing:** Build and update a SQLite-backed repository index.
- **Semantic code search:** Retrieve relevant code using Gemini embeddings and hybrid semantic/lexical ranking.
- **Context-aware retrieval:** Use query intent, contextual ranking, result diversification, confidence signals, and context packing.
- **Retrieval caching:** Cache retrieval results and inspect cache statistics.
- **AI agent integration:** An example Gemini-powered agent uses MCP tools to answer repository-specific questions.
- **Testing and evaluation:** Automated tests and retrieval-evaluation support help validate functionality and retrieval quality.
- **Configuration and safety:** Validate runtime configuration, restrict repository file access, and apply file-size limits.

## Architecture

```text
Developer / AI Coding Agent
            |
            v
        MCP Client
            |
            v
      RepoMind Server
       (FastMCP)
            |
     +------+------+
     |      |      |
     v      v      v
 Repository  Code   Git
  Tools    Index   Tools
     |      |
     |      v
     |   SQLite Index
     |      |
     +------+
            |
            v
  Hybrid Semantic Retrieval
    + Gemini Embeddings
            |
            v
  Ranked, Relevant Code Context
```

## Tech Stack

- Python 3.11+
- Model Context Protocol (MCP)
- FastMCP
- SQLite
- Python AST
- Google Gen AI SDK
- Gemini embeddings
- pytest

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Tusharmakholiya10/repo-intelligence-mcp.git
cd repo-intelligence-mcp
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python documentation for virtual-environment activation options or invoke the environment's Python executable directly.

### 3. Install RepoMind

Install the project and its development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

This installs the runtime dependencies, including `google-genai`, and pytest for development.

### 4. Verify the installation

```powershell
python -c "import repomind.server; print('RepoMind import successful')"
python -m pytest -q
```

The MCP server is a long-running process. The `repomind` command starts the server; it is not a conventional command-line help utility.

## Configuration

RepoMind reads its runtime configuration from environment variables.

| Variable                   | Required?                    | Default            | Purpose                                     |
| -------------------------- | ---------------------------- | ------------------ | ------------------------------------------- |
| `REPOMIND_REPO`            | No                           | Current directory  | Root directory of the repository to inspect |
| `GEMINI_API_KEY`           | For semantic indexing/search | None               | Authenticates Gemini API requests           |
| `GEMINI_MODEL`             | No                           | `gemini-3.8-flash` | Model used by the example interactive agent |
| `REPOMIND_MAX_FILE_SIZE`   | No                           | `1000000`          | Maximum readable file size in bytes         |
| `REPOMIND_HTTP_TIMEOUT_MS` | No                           | `30000`            | Timeout for individual Gemini HTTP requests |

The repository path must exist and point to a directory. File-size and timeout settings are validated against configured limits.

Basic repository tools do not require a Gemini API key. Semantic indexing and semantic search require `GEMINI_API_KEY`.

Set the variables in your MCP client's environment configuration or in the shell that launches RepoMind. Do not commit API keys, credentials, or populated environment files to Git.

## Running RepoMind

After installation, start the MCP server with:

```powershell
$env:REPOMIND_REPO = "C:\path\to\your\repository"
repomind
```

Replace the example path with the repository you want RepoMind to analyze. The server runs until its process is stopped.

To use semantic indexing and search, configure your Gemini API key before starting the server:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Use a real key only in your local environment or a secure secret manager. Never commit it to the repository.

## MCP Client Configuration

MCP clients that support local **stdio servers** can launch RepoMind as a subprocess. The following is an illustrative JSON configuration; adapt the location and configuration format to your MCP client.

```json
{
  "mcpServers": {
    "repomind": {
      "command": "C:\\path\\to\\repo-intelligence-mcp\\.venv\\Scripts\\repomind.exe",
      "args": [],
      "env": {
        "REPOMIND_REPO": "C:\\path\\to\\your\\target-repository",
        "GEMINI_API_KEY": "YOUR_GEMINI_API_KEY"
      }
    }
  }
}
```

Replace both paths and provide the API key securely if you intend to use semantic features. If you use a different virtual-environment directory, update the executable path accordingly. Basic repository tools can be used without the Gemini key.

## Available MCP Tools

RepoMind currently registers 15 tools.

| Tool                    | Purpose                                           |
| ----------------------- | ------------------------------------------------- |
| `list_files`            | Explore repository files                          |
| `read_file`             | Read a repository file                            |
| `search_code`           | Search source code and text                       |
| `analyze_python_file`   | Analyze Python structure and imports              |
| `search_symbols`        | Search indexed symbols                            |
| `find_usages`           | Find references to a symbol                       |
| `get_dependencies`      | Inspect local file dependencies                   |
| `index_repository`      | Build or update the repository index              |
| `index_stats`           | Inspect index statistics                          |
| `semantic_search`       | Search for relevant code using semantic retrieval |
| `build_context`         | Assemble compact, agent-ready code context        |
| `get_git_history`       | Retrieve recent commits                           |
| `get_git_commit`        | Inspect a specific commit                         |
| `get_git_file_history`  | Inspect a file's commit history                   |
| `retrieval_cache_stats` | Inspect retrieval-cache statistics                |

## Example Questions

With RepoMind connected to an AI coding agent, you can ask questions such as:

- "What does `repository.py` depend on?"
- "Where is `PythonAnalyzer` defined and used?"
- "Which files import this module?"
- "Explain how authentication is implemented in this repository."
- "Find the code responsible for validating file paths."
- "What changed in the latest commit?"
- "Build a concise context bundle for this function."
- "Find code related to database connection handling."

The agent should use the appropriate RepoMind tools to gather repository evidence before answering code-specific questions.

## Example Gemini Agent

The repository includes `examples/agent.py`, which demonstrates an interactive Gemini-powered agent using RepoMind through MCP.

Configure `GEMINI_API_KEY`, `REPOMIND_REPO`, and any desired model settings before running it from the repository's root directory:

```powershell
python .\examples\agent.py
```

The example uses MCP tool calls and retrieval strategies to help answer repository questions. Gemini API usage may incur charges according to your Google AI account and applicable pricing.

## Testing

Run the automated test suite:

```powershell
python -m pytest -q
```

The tests cover repository operations, analysis, indexing, dependency resolution, retrieval behavior, configuration, and related regression scenarios.

## Project Structure

```text
repo-intelligence-mcp/
├── src/
│   └── repomind/
│       ├── server.py
│       ├── repository.py
│       ├── analyzer.py
│       ├── indexer.py
│       ├── embeddings.py
│       ├── chunker.py
│       ├── agentic_retrieval.py
│       ├── query_intent.py
│       ├── query_reformulation.py
│       ├── retrieval_cache.py
│       ├── retrieval_diversification.py
│       ├── retrieval_explanation.py
│       ├── retrieval_strategy.py
│       ├── context_packer.py
│       ├── git.py
│       ├── config.py
│       └── roles.py
├── examples/
│   └── agent.py
├── evaluation/
├── tests/
├── pyproject.toml
└── README.md
```

## Development Status

Implemented capabilities include MCP-based repository tools, AST analysis, symbol and usage search, dependency indexing, Git intelligence, incremental indexing, Gemini embeddings, hybrid retrieval, context packing, and retrieval evaluation.

The project is under active development. Tool behavior, retrieval quality, and compatibility with specific MCP clients should be validated in the target environment.

## Contributing

1. Create a branch for your change.
2. Add or update tests for changed behavior.
3. Run `python -m pytest -q`.
4. Review the changes with `git diff --check` and `git diff`.
5. Submit a pull request describing the change and its validation.

## License

Add a license file and update this section if you intend to distribute RepoMind under an open-source license.
