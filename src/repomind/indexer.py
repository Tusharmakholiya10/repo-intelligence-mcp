import sqlite3
from pathlib import Path
from datetime import datetime


class CodeIndexer:
    """Persistent SQLite index for repository code intelligence."""

    def __init__(
        self,
        repository_root: Path,
        database_path: Path | None = None,
    ):
        self.repository_root = repository_root.resolve()

        if database_path is None:
            database_path = (
                self.repository_root
                / ".repomind"
                / "index.db"
            )

        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(self):
        """Create a SQLite database connection."""

        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize_database(self):
        """Create database tables if they do not exist."""

        with self._connect() as connection:

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    language TEXT,
                    size INTEGER NOT NULL,
                    modified_time TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,

                    FOREIGN KEY(file_id)
                        REFERENCES files(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_symbols_name
                    ON symbols(name);

                CREATE INDEX IF NOT EXISTS
                    idx_symbols_qualified_name
                    ON symbols(qualified_name);

                CREATE INDEX IF NOT EXISTS
                    idx_symbols_file_id
                    ON symbols(file_id);

                CREATE TABLE IF NOT EXISTS symbol_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    symbol_name TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    reference_type TEXT NOT NULL,

                    FOREIGN KEY(file_id)
                        REFERENCES files(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_references_symbol
                    ON symbol_references(symbol_name);
                """
            )

    def index_file(
        self,
        relative_path: str,
        language: str,
        size: int,
        modified_time: str,
        symbols: list[dict],
    ):
        """Insert or replace a file and its symbols."""

        with self._connect() as connection:

            cursor = connection.execute(
                """
                INSERT INTO files (
                    path,
                    language,
                    size,
                    modified_time
                )
                VALUES (?, ?, ?, ?)

                ON CONFLICT(path)
                DO UPDATE SET
                    language = excluded.language,
                    size = excluded.size,
                    modified_time = excluded.modified_time
                """,
                (
                    relative_path,
                    language,
                    size,
                    modified_time,
                ),
            )

            file_row = connection.execute(
                """
                SELECT id
                FROM files
                WHERE path = ?
                """,
                (relative_path,),
            ).fetchone()

            file_id = file_row["id"]

            connection.execute(
                """
                DELETE FROM symbols
                WHERE file_id = ?
                """,
                (file_id,),
            )

            for symbol in symbols:

                connection.execute(
                    """
                    INSERT INTO symbols (
                        file_id,
                        name,
                        qualified_name,
                        type,
                        line,
                        end_line
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        symbol["name"],
                        symbol["qualified_name"],
                        symbol["type"],
                        symbol["line"],
                        symbol["end_line"],
                    ),
                )

            connection.commit()

    def search_symbols(
        self,
        query: str,
        max_results: int = 100,
    ) -> list[dict]:
        """Search indexed symbols."""

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT
                    files.path,
                    symbols.name,
                    symbols.qualified_name,
                    symbols.type,
                    symbols.line,
                    symbols.end_line
                FROM symbols
                JOIN files
                    ON symbols.file_id = files.id
                WHERE
                    symbols.name LIKE ?
                    OR symbols.qualified_name LIKE ?
                ORDER BY files.path, symbols.line
                LIMIT ?
                """,
                (
                    f"%{query}%",
                    f"%{query}%",
                    max_results,
                ),
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

    def get_stats(self) -> dict:
        """Return basic index statistics."""

        with self._connect() as connection:

            file_count = connection.execute(
                "SELECT COUNT(*) FROM files"
            ).fetchone()[0]

            symbol_count = connection.execute(
                "SELECT COUNT(*) FROM symbols"
            ).fetchone()[0]

            reference_count = connection.execute(
                "SELECT COUNT(*) FROM symbol_references"
            ).fetchone()[0]

        return {
            "files": file_count,
            "symbols": symbol_count,
            "references": reference_count,
            "database": str(self.database_path),
        }