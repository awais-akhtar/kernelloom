"""Virtual model and KV-cache memory management primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import threading
import time
from typing import Any, Iterable


class MemoryTier(StrEnum):
    ACCELERATOR = "accelerator"
    SHARED_RAM = "shared_ram"
    PAGEABLE_RAM = "pageable_ram"
    STORAGE = "storage"


@dataclass
class KVBlock:
    id: str
    token_hash: str
    token_count: int
    size_bytes: int
    tier: MemoryTier
    ref_count: int
    last_access: float
    dirty: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tier"] = self.tier.value
        return payload


@dataclass
class KVSession:
    id: str
    block_ids: list[str]
    priority: int
    created_at: float
    last_access: float
    token_count: int = 0


class VirtualKVCache:
    """Thread-safe paged KV metadata manager with shared-prefix COW semantics.

    Tensor payload allocation remains the backend's responsibility. This class
    owns block identity, capacity accounting, sharing, and deterministic
    eviction, which are the parts that must remain backend-independent.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        block_tokens: int = 16,
        bytes_per_token: int = 256 * 1024,
        strategy: str = "paged",
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if block_tokens <= 0 or bytes_per_token <= 0:
            raise ValueError("block_tokens and bytes_per_token must be positive")
        if strategy not in {"paged", "virtual-contiguous", "sliding-window"}:
            raise ValueError("unsupported KV strategy")
        self.capacity_bytes = int(capacity_bytes)
        self.block_tokens = int(block_tokens)
        self.bytes_per_token = int(bytes_per_token)
        self.strategy = strategy
        self.blocks: dict[str, KVBlock] = {}
        self.sessions: dict[str, KVSession] = {}
        self.prefix_index: dict[str, str] = {}
        self._lock = threading.RLock()

    def create_session(
        self,
        session_id: str,
        *,
        prefix_tokens: Iterable[int] = (),
        priority: int = 50,
    ) -> dict[str, Any]:
        session_id = str(session_id).strip()
        if not session_id:
            raise ValueError("session_id is required")
        with self._lock:
            if session_id in self.sessions:
                raise ValueError("session already exists")
            now = time.monotonic()
            session = KVSession(session_id, [], max(0, min(int(priority), 100)), now, now)
            self.sessions[session_id] = session
            tokens = [int(value) for value in prefix_tokens]
            for offset in range(0, len(tokens), self.block_tokens):
                chunk = tokens[offset : offset + self.block_tokens]
                digest = _token_hash(chunk)
                existing_id = self.prefix_index.get(digest)
                existing = self.blocks.get(existing_id or "")
                if existing and existing.token_count == len(chunk):
                    existing.ref_count += 1
                    existing.last_access = now
                    session.block_ids.append(existing.id)
                else:
                    block = self._allocate_block(chunk, dirty=False)
                    self.prefix_index[digest] = block.id
                    session.block_ids.append(block.id)
                session.token_count += len(chunk)
            self._evict_if_needed(protected={session_id})
            return self.session_status(session_id)

    def append(self, session_id: str, token_ids: Iterable[int]) -> dict[str, Any]:
        tokens = [int(value) for value in token_ids]
        if not tokens:
            return self.session_status(session_id)
        with self._lock:
            session = self._session(session_id)
            now = time.monotonic()
            while tokens:
                tail = self.blocks.get(session.block_ids[-1]) if session.block_ids else None
                if tail and tail.token_count < self.block_tokens:
                    if tail.ref_count > 1:
                        tail.ref_count -= 1
                        replacement = self._allocate_block([], dirty=True)
                        replacement.token_count = tail.token_count
                        replacement.size_bytes = tail.size_bytes
                        replacement.token_hash = tail.token_hash
                        session.block_ids[-1] = replacement.id
                        tail = replacement
                    count = min(self.block_tokens - tail.token_count, len(tokens))
                    consumed = tokens[:count]
                    tokens = tokens[count:]
                    tail.token_count += count
                    tail.size_bytes = tail.token_count * self.bytes_per_token
                    tail.token_hash = hashlib.sha256((tail.token_hash + _token_hash(consumed)).encode("ascii")).hexdigest()
                    tail.dirty = True
                    tail.last_access = now
                    session.token_count += count
                else:
                    consumed = tokens[: self.block_tokens]
                    tokens = tokens[self.block_tokens :]
                    block = self._allocate_block(consumed, dirty=True)
                    session.block_ids.append(block.id)
                    session.token_count += len(consumed)
            session.last_access = now
            self._evict_if_needed(protected={session_id})
            return self.session_status(session_id)

    def release(self, session_id: str) -> bool:
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if not session:
                return False
            for block_id in session.block_ids:
                block = self.blocks.get(block_id)
                if not block:
                    continue
                block.ref_count -= 1
                if block.ref_count <= 0:
                    self._drop_block(block_id)
            return True

    def evict(self, *, bytes_required: int = 0, exclude_sessions: Iterable[str] = ()) -> dict[str, Any]:
        with self._lock:
            before = self.used_bytes
            target = max(0, int(bytes_required))
            excluded = set(exclude_sessions)
            candidates = sorted(
                (session for session in self.sessions.values() if session.id not in excluded),
                key=lambda item: (item.priority, item.last_access),
            )
            evicted: list[str] = []
            for session in candidates:
                if target and before - self.used_bytes >= target:
                    break
                evicted.append(session.id)
                self.release(session.id)
            return {"evicted_sessions": evicted, "freed_bytes": before - self.used_bytes}

    @property
    def used_bytes(self) -> int:
        return sum(block.size_bytes for block in self.blocks.values())

    def session_status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            session.last_access = time.monotonic()
            return {
                "id": session.id,
                "token_count": session.token_count,
                "block_count": len(session.block_ids),
                "shared_blocks": sum(1 for block_id in session.block_ids if self.blocks[block_id].ref_count > 1),
                "priority": session.priority,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            shared = sum(1 for block in self.blocks.values() if block.ref_count > 1)
            return {
                "strategy": self.strategy,
                "capacity_bytes": self.capacity_bytes,
                "used_bytes": self.used_bytes,
                "utilization": round(self.used_bytes / self.capacity_bytes, 6),
                "block_tokens": self.block_tokens,
                "bytes_per_token": self.bytes_per_token,
                "blocks": len(self.blocks),
                "shared_blocks": shared,
                "sessions": len(self.sessions),
            }

    def _allocate_block(self, tokens: list[int], *, dirty: bool) -> KVBlock:
        digest = _token_hash(tokens)
        identity = hashlib.sha256(f"{digest}:{time.monotonic_ns()}:{len(self.blocks)}".encode("ascii")).hexdigest()[:24]
        block = KVBlock(
            id=identity,
            token_hash=digest,
            token_count=len(tokens),
            size_bytes=len(tokens) * self.bytes_per_token,
            tier=MemoryTier.ACCELERATOR,
            ref_count=1,
            last_access=time.monotonic(),
            dirty=dirty,
        )
        self.blocks[identity] = block
        return block

    def _evict_if_needed(self, *, protected: set[str]) -> None:
        overflow = self.used_bytes - self.capacity_bytes
        if overflow <= 0:
            return
        result = self.evict(bytes_required=overflow, exclude_sessions=protected)
        if self.used_bytes > self.capacity_bytes:
            raise MemoryError(
                f"KV cache capacity exceeded by {self.used_bytes - self.capacity_bytes} bytes; "
                f"evicted {len(result['evicted_sessions'])} lower-priority sessions"
            )

    def _drop_block(self, block_id: str) -> None:
        block = self.blocks.pop(block_id, None)
        if block and self.prefix_index.get(block.token_hash) == block_id:
            self.prefix_index.pop(block.token_hash, None)

    def _session(self, session_id: str) -> KVSession:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"unknown KV session: {session_id}")
        return session


def _token_hash(tokens: Iterable[int]) -> str:
    hasher = hashlib.sha256()
    for token in tokens:
        hasher.update(int(token).to_bytes(8, "little", signed=True))
    return hasher.hexdigest()
