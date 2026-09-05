import sqlite3
import struct
from pathlib import Path


class CodeIndexer:
    """Persistent SQLite index for repository code intelligence."""

    @staticmethod
    def _serialize_embedding(
        embedding: list[float],
    ) -> bytes:
        """Serialize a float vector into SQLite-compatible bytes."""

        if not embedding:
            raise ValueError(
                "Embedding cannot be empty."
            )

        return struct.pack(
            f"<{len(embedding)}d",
            *embedding,
        )

    @staticmethod
    def _deserialize_embedding(
        data: bytes,
    ) -> list[float]:
        """Deserialize SQLite embedding bytes into floats."""

        if not data:
            raise ValueError(
                "Embedding data cannot be empty."
            )

        float_count = len(data) // 8

        return list(
            struct.unpack(
                f"<{float_count}d",
                data,
            )
        )



    def index_semantic_chunks(
        self,
        relative_path: str,
        chunks: list[dict],
    ):
        """
        Replace semantic chunks for a repository file.
        
        """

        relative_path = self._normalize_path(
            relative_path
        )

        with self._connect() as connection:

            file_row = connection.execute(
                """
                SELECT id
                FROM files
                WHERE path = ?
                """,
                (relative_path,),
            ).fetchone()

            if file_row is None:
                raise ValueError(
                    f"File is not indexed: {relative_path}"
                )

            file_id = file_row["id"]

            connection.execute(
                """
                DELETE FROM semantic_chunks
                WHERE file_id = ?
                """,
                (file_id,),
            )

            for chunk in chunks:

                embedding = chunk.get(
                    "embedding"
                )

                if not embedding:
                    raise ValueError(
                        "Semantic chunk is missing embedding."
                    )

                embedding_blob = (
                    self._serialize_embedding(
                        embedding
                    )
                )

                connection.execute(
                    """
                    INSERT INTO semantic_chunks (
                        file_id,
                        start_line,
                        end_line,
                        symbol_name,
                        symbol_type,
                        content,
                        embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        chunk["start_line"],
                        chunk["end_line"],
                        chunk.get("symbol_name"),
                        chunk.get("symbol_type"),
                        chunk["content"],
                        embedding_blob,
                    ),
                )

            connection.commit()

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

        # Required for ON DELETE CASCADE to work in SQLite.
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    def _normalize_path(self, relative_path: str) -> str:
        """
        Normalize repository-relative paths.

        SQLite stores paths using forward slashes so that
        Windows and Unix-style paths are treated consistently.
        """

        return str(
            Path(relative_path)
        ).replace("\\", "/")

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

                CREATE TABLE IF NOT EXISTS semantic_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                symbol_name TEXT,
                symbol_type TEXT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,

                FOREIGN KEY(file_id)
                    REFERENCES files(id)
                    ON DELETE CASCADE
            );
                
                CREATE INDEX IF NOT EXISTS
                idx_semantic_chunks_file_id
                ON semantic_chunks(file_id);
            
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

                CREATE TABLE IF NOT EXISTS dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file_id INTEGER NOT NULL,
                    target_file_id INTEGER NOT NULL,
                    dependency_type TEXT NOT NULL,
                    line INTEGER,

                    UNIQUE(
                        source_file_id,
                        target_file_id,
                        dependency_type,
                        line
                    ),

                    FOREIGN KEY(source_file_id)
                        REFERENCES files(id)
                        ON DELETE CASCADE,

                    FOREIGN KEY(target_file_id)
                        REFERENCES files(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_dependencies_source
                    ON dependencies(source_file_id);

                CREATE INDEX IF NOT EXISTS
                    idx_dependencies_target
                    ON dependencies(target_file_id);
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

        relative_path = self._normalize_path(
            relative_path
        )

        with self._connect() as connection:

            connection.execute(
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

    def needs_reindex(
        self,
        relative_path: str,
        size: int,
        modified_time: str,
    ) -> bool:
        """
        Return True if a file is new or has changed
        since it was last indexed.
        """

        relative_path = self._normalize_path(
            relative_path
        )

        with self._connect() as connection:

            row = connection.execute(
                """
                SELECT size, modified_time
                FROM files
                WHERE path = ?
                """,
                (relative_path,),
            ).fetchone()

            # File is not indexed yet.
            if row is None:
                return True

            # File size changed.
            if row["size"] != size:
                return True

            # Modification time changed.
            if row["modified_time"] != modified_time:
                return True

            return False

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

            dependency_count = connection.execute(
                "SELECT COUNT(*) FROM dependencies"
            ).fetchone()[0]

            semantic_chunk_count = connection.execute(
                "SELECT COUNT(*) FROM semantic_chunks"
            ).fetchone()[0]

        return {
            "files": file_count,
            "symbols": symbol_count,
            "references": reference_count,
            "dependencies": dependency_count,
            "semantic_chunks": semantic_chunk_count,
            "database": str(self.database_path),
        }

    def index_references(
        self,
        relative_path: str,
        references: list[dict],
    ):
        """Replace references for a file."""

        relative_path = self._normalize_path(
            relative_path
        )

        with self._connect() as connection:

            file_row = connection.execute(
                """
                SELECT id
                FROM files
                WHERE path = ?
                """,
                (relative_path,),
            ).fetchone()

            if file_row is None:
                raise ValueError(
                    f"File is not indexed: {relative_path}"
                )

            file_id = file_row["id"]

            connection.execute(
                """
                DELETE FROM symbol_references
                WHERE file_id = ?
                """,
                (file_id,),
            )

            for reference in references:

                connection.execute(
                    """
                    INSERT INTO symbol_references (
                        file_id,
                        symbol_name,
                        line,
                        reference_type
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        reference["symbol_name"],
                        reference["line"],
                        reference["reference_type"],
                    ),
                )

            connection.commit()

    def find_usages(
        self,
        symbol_name: str,
        max_results: int = 100,
    ) -> list[dict]:
        """Find indexed references to a symbol."""

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT
                    files.path,
                    symbol_references.symbol_name,
                    symbol_references.line,
                    symbol_references.reference_type
                FROM symbol_references
                JOIN files
                    ON symbol_references.file_id = files.id
                WHERE symbol_references.symbol_name = ?
                ORDER BY
                    files.path,
                    symbol_references.line
                LIMIT ?
                """,
                (
                    symbol_name,
                    max_results,
                ),
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

    def index_dependencies(
        self,
        relative_path: str,
        dependencies: list[dict],
    ):
        """
        Store local file dependencies for a source file.
        """

        relative_path = self._normalize_path(
            relative_path
        )

        with self._connect() as connection:

            source_row = connection.execute(
                """
                SELECT id
                FROM files
                WHERE path = ?
                """,
                (relative_path,),
            ).fetchone()

            if source_row is None:
                raise ValueError(
                    f"File is not indexed: {relative_path}"
                )

            source_id = source_row["id"]

            connection.execute(
                """
                DELETE FROM dependencies
                WHERE source_file_id = ?
                """,
                (source_id,),
            )

            for dependency in dependencies:

                target_file = self._normalize_path(
                    dependency["target_file"]
                )

                target_row = connection.execute(
                    """
                    SELECT id
                    FROM files
                    WHERE path = ?
                    """,
                    (target_file,),
                ).fetchone()

                if target_row is None:
                    continue

                connection.execute(
                    """
                    INSERT OR IGNORE INTO dependencies (
                        source_file_id,
                        target_file_id,
                        dependency_type,
                        line
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        target_row["id"],
                        dependency["type"],
                        dependency["line"],
                    ),
                )

            connection.commit()

    def get_dependencies(
        self,
        relative_path: str,
        max_results: int = 100,
    ) -> list[dict]:
        """
        Return files that a source file depends on.
        """

        relative_path = self._normalize_path(
            relative_path
        )

        with self._connect() as connection:

            rows = connection.execute(
                """
                SELECT
                    target.path,
                    dependencies.dependency_type,
                    dependencies.line
                FROM dependencies
                JOIN files AS source
                    ON dependencies.source_file_id = source.id
                JOIN files AS target
                    ON dependencies.target_file_id = target.id
                WHERE source.path = ?
                ORDER BY dependencies.line
                LIMIT ?
                """,
                (
                    relative_path,
                    max_results,
                ),
            ).fetchall()

            return [
                dict(row)
                for row in rows
            ]

    def get_semantic_chunks(
        self,
        relative_path: str | None = None,
    ) -> list[dict]:
        """Return stored semantic chunks."""

        with self._connect() as connection:

            if relative_path is None:

                rows = connection.execute(
                    """
                    SELECT
                        files.path,
                        semantic_chunks.start_line,
                        semantic_chunks.end_line,
                        semantic_chunks.symbol_name,
                        semantic_chunks.symbol_type,
                        semantic_chunks.content,
                        semantic_chunks.embedding
                    FROM semantic_chunks
                    JOIN files
                        ON semantic_chunks.file_id = files.id
                    ORDER BY
                        files.path,
                        semantic_chunks.start_line
                    """
                ).fetchall()

            else:

                relative_path = (
                    self._normalize_path(
                        relative_path
                    )
                )

                rows = connection.execute(
                    """
                    SELECT
                        files.path,
                        semantic_chunks.start_line,
                        semantic_chunks.end_line,
                        semantic_chunks.symbol_name,
                        semantic_chunks.symbol_type,
                        semantic_chunks.content,
                        semantic_chunks.embedding
                    FROM semantic_chunks
                    JOIN files
                        ON semantic_chunks.file_id = files.id
                    WHERE files.path = ?
                    ORDER BY
                        semantic_chunks.start_line
                    """,
                    (relative_path,),
                ).fetchall()

            results = []

            for row in rows:

                result = dict(row)

                result["embedding"] = (
                    self._deserialize_embedding(
                        row["embedding"]
                    )
                )

                results.append(
                    result
                )

            return results