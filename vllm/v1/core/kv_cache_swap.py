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

A parallel content-hash index (`_completed_by_hash`) supports paper §6.5
cross-request reuse: when a swap-out completes and its block hash is
known, the (hash → path) mapping is registered so subsequent requests
that compute the same chained block hash during prefill can recover the
K/V bytes instead of re-prefilling.

This is a pure data structure — no I/O. The eviction manager and
scheduler drive transitions via `pin`, `complete`, `drop_request`,
`lookup_by_hash`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True, slots=True)
class PendingSwap:
    """A swap-out in flight: GPU block kept allocated until ack.

    ``block_hash`` is the bare (non-group-qualified) chained block hash
    captured at eviction time; carried through to ``complete`` so the
    hash index can be populated on ack.
    """
    group_idx: int
    block_id: int
    block_hash: BlockHash | None = None


@dataclass(slots=True)
class SwapStore:
    # req_id -> {block_pos -> PendingSwap}
    _pending: dict[str, dict[int, PendingSwap]] = (
        field(default_factory=dict)
    )
    # req_id -> {block_pos -> on-disk path}
    _completed: dict[str, dict[int, str]] = (
        field(default_factory=dict)
    )
    # block_hash -> on-disk path. Parallel index for cross-request
    # lookup (paper §6.5). Populated on ``complete`` when the hash is
    # known; cleaned on ``consume`` and on ``drop_request``.
    _completed_by_hash: dict[BlockHash, str] = (
        field(default_factory=dict)
    )
    # block_hash -> (req_id, block_pos). Reverse map so that
    # ``consume`` and ``drop_request`` can clean ``_completed_by_hash``
    # without walking it.
    _hash_owner: dict[BlockHash, tuple[str, int]] = (
        field(default_factory=dict)
    )

    def pin(
        self,
        req_id: str,
        block_pos: int,
        group_idx: int,
        block_id: int,
        block_hash: BlockHash | None = None,
    ) -> None:
        """Mark a block as queued for swap-out; GPU id is held.

        ``block_hash`` is the bare (non-group-qualified) chained content
        hash for cross-request reuse (paper §6.5). May be None for
        decode-generated blocks that haven't been hashed yet — those
        entries remain reachable via the (req_id, block_pos) index but
        are not cross-request reusable.
        """
        self._pending.setdefault(req_id, {})[block_pos] = PendingSwap(
            group_idx=group_idx,
            block_id=block_id,
            block_hash=block_hash,
        )

    def complete(
        self,
        req_id: str,
        block_pos: int,
        path: str,
    ) -> PendingSwap | None:
        """Move pending → completed. Returns the freed PendingSwap, or
        None if the entry was already dropped (e.g., request finished
        between directive and ack).

        If the pending entry carried a ``block_hash``, also registers
        the (hash → path) mapping for cross-request lookup.
        """
        pending = self._pending.get(req_id)
        if pending is None:
            return None
        entry = pending.pop(block_pos, None)
        if entry is None:
            return None
        if not pending:
            self._pending.pop(req_id, None)
        self._completed.setdefault(req_id, {})[block_pos] = path
        if entry.block_hash is not None:
            self._completed_by_hash[entry.block_hash] = path
            self._hash_owner[entry.block_hash] = (req_id, block_pos)
        return entry

    def lookup(self, req_id: str, block_pos: int) -> str | None:
        """Path on disk if the block is swapped out and complete."""
        per_req = self._completed.get(req_id)
        return per_req.get(block_pos) if per_req is not None else None

    def lookup_by_hash(self, block_hash: BlockHash) -> str | None:
        """Path on disk if a block with this content hash is swapped
        out and complete (paper §6.5 cross-request lookup)."""
        return self._completed_by_hash.get(block_hash)

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
        # Walk the reverse owner map to drop any matching hash entry.
        # Cheap: there's at most one (hash, owner) pair per (req, pos).
        stale_hashes = [
            h for h, owner in self._hash_owner.items()
            if owner == (req_id, block_pos)
        ]
        for h in stale_hashes:
            self._hash_owner.pop(h, None)
            self._completed_by_hash.pop(h, None)
        return path

    def drop_request(
        self, req_id: str
    ) -> tuple[list[PendingSwap], list[str]]:
        """Drop all entries for a finished request.

        Returns (still_pending, completed_paths) so the caller can free
        the pinned GPU blocks and unlink the disk files. Also clears
        the parallel content-hash index for blocks owned by this
        request — used when ``keep_swap_on_finish=False`` (default).
        """
        pending = list(self._pending.pop(req_id, {}).values())
        paths = list(self._completed.pop(req_id, {}).values())
        # Drop any hash-index entries owned by this request.
        stale_hashes = [
            h for h, owner in self._hash_owner.items()
            if owner[0] == req_id
        ]
        for h in stale_hashes:
            self._hash_owner.pop(h, None)
            self._completed_by_hash.pop(h, None)
        return pending, paths

    def drop_completed_only(self, req_id: str) -> list[str]:
        """Drop only completed entries; preserve pending.

        Used by the finish-dump path (paper §6.5) when the user wants
        the swap tier to retain only the resident working set, not the
        cold periphery that Otsu evicted during the request. Pending
        swap-outs from Otsu (still in flight) are left alone — they'll
        complete naturally and re-populate ``_completed``, but those
        entries reflect cold blocks; if the user wants a strict
        hot-only invariant they should run finish-dump only after all
        prior pending acks have landed.

        Returns the list of on-disk paths the caller should unlink.
        """
        paths = list(self._completed.pop(req_id, {}).values())
        # Drop hash-index entries owned by this request.
        stale_hashes = [
            h for h, owner in self._hash_owner.items()
            if owner[0] == req_id
        ]
        for h in stale_hashes:
            self._hash_owner.pop(h, None)
            self._completed_by_hash.pop(h, None)
        return paths

    def drop_request_pending_only(
        self, req_id: str
    ) -> list[PendingSwap]:
        """Drop only pending swap-out entries; preserve completed.

        Used when ``keep_swap_on_finish=True`` (paper §6.5): the
        finished request's GPU blocks must be released, but completed
        swap-out entries (already on disk) stay in the SwapStore for
        cross-request hash lookups by subsequent requests.
        """
        return list(self._pending.pop(req_id, {}).values())

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
