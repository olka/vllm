# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side table for disk-backed KV cache swap.

Owned by the scheduler-side `KVCacheEvictionManager`. Tracks two states
per evicted block:

- *pending*: swap-out directive sent to worker; GPU block id is pinned
  (NOT yet returned to the pool) until the worker acks completion.
- *completed*: KV bytes are on disk; GPU block id has been freed.
  Path is retained so a future cache miss can swap the block back in
  instead of recomputing it.

This is a pure data structure — no I/O. The eviction manager and
scheduler drive transitions via `pin`, `complete`, `drop_request`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PendingSwap:
    """A swap-out in flight: GPU block kept allocated until ack."""
    group_idx: int
    block_id: int


@dataclass(slots=True)
class SwapStore:
    # req_id -> {block_pos -> PendingSwap}
    _pending: dict[str, dict[int, PendingSwap]]
    # req_id -> {block_pos -> on-disk path}
    _completed: dict[str, dict[int, str]]

    def __init__(self) -> None:
        self._pending = {}
        self._completed = {}

    def pin(
        self,
        req_id: str,
        block_pos: int,
        group_idx: int,
        block_id: int,
    ) -> None:
        """Mark a block as queued for swap-out; GPU id is held."""
        self._pending.setdefault(req_id, {})[block_pos] = PendingSwap(
            group_idx=group_idx, block_id=block_id
        )

    def complete(
        self,
        req_id: str,
        block_pos: int,
        path: str,
    ) -> PendingSwap | None:
        """Move pending → completed. Returns the freed PendingSwap, or
        None if the entry was already dropped (e.g., request finished
        between directive and ack)."""
        pending = self._pending.get(req_id)
        if pending is None:
            return None
        entry = pending.pop(block_pos, None)
        if entry is None:
            return None
        if not pending:
            self._pending.pop(req_id, None)
        self._completed.setdefault(req_id, {})[block_pos] = path
        return entry

    def lookup(self, req_id: str, block_pos: int) -> str | None:
        """Path on disk if the block is swapped out and complete."""
        per_req = self._completed.get(req_id)
        return per_req.get(block_pos) if per_req is not None else None

    def consume(self, req_id: str, block_pos: int) -> str | None:
        """Drop a single completed entry that has been swapped back in.

        Returns the path that was associated with the entry (caller may
        choose to unlink the file, though for a successful swap-in the
        bytes are already on the GPU and the file is no longer needed).
        Returns None if the entry was not present (already consumed,
        never existed, or request was dropped).
        """
        per_req = self._completed.get(req_id)
        if per_req is None:
            return None
        path = per_req.pop(block_pos, None)
        if not per_req:
            self._completed.pop(req_id, None)
        return path

    def drop_request(
        self, req_id: str
    ) -> tuple[list[PendingSwap], list[str]]:
        """Drop all entries for a finished request.

        Returns (still_pending, completed_paths) so the caller can free
        the pinned GPU blocks and unlink the disk files.
        """
        pending = list(self._pending.pop(req_id, {}).values())
        paths = list(self._completed.pop(req_id, {}).values())
        return pending, paths

    def has_pending(self, req_id: str | None = None) -> bool:
        if req_id is None:
            return bool(self._pending)
        return req_id in self._pending

    def is_pending(self, req_id: str, block_pos: int) -> bool:
        """True iff swap-out for (req_id, block_pos) is in flight —
        directive queued, worker ack not yet received. Distinguishes
        the "wait one cycle" case from a genuine ``lookup`` miss in
        the page-fault handler."""
        per_req = self._pending.get(req_id)
        return per_req is not None and block_pos in per_req
