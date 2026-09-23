from __future__ import annotations

import re


class ComponentRoleClassifier:
    """Classify repository components into deterministic architecture roles."""

    ROLES = {
        "security",
        "repository_access",
        "chunking",
        "embedding",
        "indexing",
        "search",
        "dependency",
        "reference_tracking",
        "orchestration",
        "agent",
        "unknown",
    }

    QUERY_ROLE_TERMS = {
        "security": {
            "security",
            "secure",
            "protected",
            "protect",
            "sensitive",
            "environment",
            "env",
            "traversal",
            "safe",
        },
        "repository_access": {
            "repository",
            "repository",
            "file",
            "files",
            "access",
            "read",
            "path",
        },
        "chunking": {
            "chunk",
            "chunks",
            "chunking",
            "chunked",
            "divided",
            "split",
            "segmented",
        },
        "embedding": {
            "embedding",
            "embeddings",
            "embed",
            "vector",
            "vectors",
            "generated",
        },
        "indexing": {
            "index",
            "indexed",
            "indexing",
            "stored",
            "store",
            "save",
            "symbols",
            "classes",
            "functions",
        },
        "search": {
            "search",
            "semantic",
            "find",
            "discover",
        },
        "dependency": {
            "dependency",
            "dependencies",
            "depend",
        },
        "reference_tracking": {
            "usage",
            "usages",
            "reference",
            "references",
        },
        "orchestration": {
            "pipeline",
            "process",
            "workflow",
            "orchestrate",
        },
        "agent": {
            "agent",
            "assistant",
            "model",
            "gemini",
        },
    }

    @classmethod
    def _tokens(cls, value: str) -> set[str]:
        """Tokenize a path, symbol, or query."""

        return {
            token
            for token in re.findall(
                r"[a-zA-Z0-9_]+",
                value.lower(),
            )
            if token
        }

    @classmethod
    def classify_component(
        cls,
        path: str,
        symbol_name: str | None = None,
    ) -> str:
        """Classify the architectural role of one repository component."""

        normalized_path = path.replace(
            "\\",
            "/",
        ).lower()

        symbol = (
            symbol_name or ""
        ).lower()

        path_name = (
            normalized_path.rsplit(
                "/",
                1,
            )[-1]
        )

        # Explicit module ownership.
        if path_name == "chunker.py":
            return "chunking"

        if path_name == "embeddings.py":
            return "embedding"

        if path_name == "repository.py":
            if any(
                term in symbol
                for term in (
                    "_resolve_safe_path",
                    "_is_ignored",
                )
            ):
                return "security"

            return "repository_access"

        if path_name == "server.py":
            if symbol == "index_repository":
                return "orchestration"

            if symbol == "semantic_search":
                return "search"

        # Symbol-specific roles inside indexer.py.
        if path_name == "indexer.py":

            if any(
                term in symbol
                for term in (
                    "find_usages",
                    "index_references",
                )
            ):
                return "reference_tracking"

            if any(
                term in symbol
                for term in (
                    "get_dependencies",
                    "index_dependencies",
                )
            ):
                return "dependency"

            if "search" in symbol:
                return "search"

            return "indexing"

        if path_name == "agent.py":
            return "agent"

        # Generic fallbacks.
        path_tokens = cls._tokens(
            normalized_path
        )
        symbol_tokens = cls._tokens(
            symbol
        )

        combined = (
            path_tokens
            | symbol_tokens
        )

        for role, terms in (
            cls.QUERY_ROLE_TERMS.items()
        ):
            if combined & terms:
                return role

        return "unknown"

    @classmethod
    def classify_query(
        cls,
        query: str,
    ) -> set[str]:
        """Return architectural roles implied by a query."""

        tokens = cls._tokens(query)

        roles = {
            role
            for role, terms
            in cls.QUERY_ROLE_TERMS.items()
            if tokens & terms
        }

        # A query involving multiple pipeline stages is also
        # an orchestration question.
        pipeline_roles = {
            "chunking",
            "embedding",
            "indexing",
        }

        if len(
            roles & pipeline_roles
        ) >= 2:
            roles.add(
                "orchestration"
            )

        return roles