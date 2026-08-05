"""SQLite persistence used by the standalone engine."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from .models import AuditEvent


class EngineStore:
    """Local state and audit storage for one or more engine projects."""

    def __init__(self, data_dir: str | Path):
        path = Path(data_dir).expanduser()
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            self.data_dir = path.parent.resolve()
            self.db_path = path.resolve()
        else:
            self.data_dir = path.resolve()
            self.db_path = self.data_dir / "kernelloom.sqlite3"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_engine_audit_project
                    ON audit_events(project, created_at DESC);
                """
            )

    def add_audit_event(self, event: AuditEvent) -> dict[str, Any]:
        with self.session() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    id, actor, action, resource, project, status, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.actor,
                    event.action,
                    event.resource,
                    event.project,
                    event.status,
                    json.dumps(event.metadata, sort_keys=True),
                    event.created_at,
                ),
            )
        return event.to_record()

    def list_audit_events(
        self,
        *,
        project: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 500))
        if project:
            query = "SELECT * FROM audit_events WHERE project = ? ORDER BY created_at DESC LIMIT ?"
            params: tuple[Any, ...] = (project, bounded_limit)
        else:
            query = "SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ?"
            params = (bounded_limit,)
        with self.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "actor": row["actor"],
                "action": row["action"],
                "resource": row["resource"],
                "project": row["project"],
                "status": row["status"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
