"""Small shared models used by the standalone engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class AuditEvent:
    actor: str
    action: str
    resource: str
    project: str = "default"
    status: str = "allowed"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "project": self.project,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
