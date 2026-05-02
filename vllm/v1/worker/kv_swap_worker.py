# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side two-tier KV swap.

Tier hierarchy:

  L1 (GPU)        — primary residence, held by the block pool.
  L2 (pinned CPU) — bounded pool of host buffers, optional. When present,
                    eviction lands here first and acks fire on D2H
                    completion (~5–15 μs per block on PCIe Gen4/5).
  L3 (NVMe disk)  — effectively unbounded. When the CPU pool is over
                    capacity, the oldest entry demotes here in a
                    background threadpool task; ack of the *new* entry
                    is not blocked on the demotion's disk write.

Disabling the CPU tier (``cpu_tier_capacity_bytes == 0``) collapses
the hierarchy to L1 + L3 — useful on hardware where host DRAM is
unified with GPU memory (GB10, MI300, Grace Hopper) and the CPU tier
offers no real capacity advantage.

The scheduler-side ``SwapStore`` is intentionally tier-agnostic: it
tracks completion of swap-out (so the GPU block id can be unpinned)
without caring which tier the bytes currently live in. Tier transitions
(CPU → NVMe under pool pressure) are managed entirely inside
``KVSwapWorker``.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class CPUSwapPool:
    """Bounded pool of pinned host buffers acting as the L2 swap tier.

    Insert evicts the oldest entry when the pool is over capacity; the
    evicted entry is returned for the caller to demote to L3 (NVMe).
    Lookup bumps LRU ordering. Per-request bulk drop is supported for
    finished-request cleanup.
    """

    def __init__(
        self, capacity_bytes: int, per_block_bytes: int
    ) -> None:
        if capacity_bytes <= 0 or per_block_bytes <= 0:
            raise ValueError(
                "capacity and per-block size must be positive"
            )
        self._max_blocks = capacity_bytes // per_block_bytes
        if self._max_blocks == 0:
            raise ValueError(
                f"CPU tier capacity ({capacity_bytes} B) is smaller "
                f"than a single block ({per_block_bytes} B)"
            )
        self._lock = threading.Lock()
        # OrderedDict gives O(1) LRU. Keys are
        # (req_id, group_idx, block_pos); values are pinned host tensors.
        self._entries: OrderedDict[
            tuple[str, int, int], torch.Tensor
        ] = OrderedDict()

    @property
    def max_blocks(self) -> int:
        return self._max_blocks

    def num_entries(self) -> int:
        with self._lock:
            return len(self._entries)

    def insert(
        self,
        key: tuple[str, int, int],
        tensor: torch.Tensor,
    ) -> tuple[tuple[str, int, int], torch.Tensor] | None:
        """Insert ``tensor`` at ``key``; bump LRU.

        Returns the evicted (oldest) entry if the pool was over
        capacity, otherwise None. The caller is responsible for
        demoting the returned entry to NVMe (or releasing it).
        """
        with self._lock:
            if key in self._entries:
                self._entries.pop(key)
            self._entries[key] = tensor
            if len(self._entries) > self._max_blocks:
                old_key, old_tensor = self._entries.popitem(last=False)
                return old_key, old_tensor
            return None

    def get(
        self, key: tuple[str, int, int]
    ) -> torch.Tensor | None:
        """Lookup with LRU bump on hit."""
        with self._lock:
            tensor = self._entries.get(key)
            if tensor is not None:
                self._entries.move_to_end(key)
            return tensor

    def remove(
        self, key: tuple[str, int, int]
    ) -> torch.Tensor | None:
        with self._lock:
            return self._entries.pop(key, None)

    def drop_request(self, req_id: str) -> int:
        """Remove all entries belonging to ``req_id``. Returns count."""
        with self._lock:
            keys_to_remove = [
                k for k in self._entries if k[0] == req_id
            ]
            for k in keys_to_remove:
                self._entries.pop(k)
            return len(keys_to_remove)


class KVSwapWorker:
    def __init__(
        self,
        swap_dir: str,
        kv_caches: list[torch.Tensor],
        cpu_tier_capacity_bytes: int = 0,
        num_io_workers: int = 4,
    ) -> None:
        if not kv_caches:
            raise ValueError("KVSwapWorker needs a non-empty kv_caches list")
        self._swap_dir = swap_dir
        self._kv_caches = kv_caches
        self._stream = torch.cuda.Stream()
        self._executor = ThreadPoolExecutor(
            max_workers=num_io_workers,
            thread_name_prefix="kv-swap-out",
        )
        self._completed: list[tuple[str, int, int, str]] = []
        # Swap-in completion/failure queues (drained per step into
        # ModelRunnerOutput.swap_in_completed / swap_in_failed).
        self._swap_in_completed: list[tuple[str, int, int]] = []
        self._swap_in_failed: list[tuple[str, int, int]] = []
        self._lock = threading.Lock()
        os.makedirs(swap_dir, exist_ok=True)
        first = kv_caches[0]
        # Per-block tensor shape on the host: [num_layers] +
        # kv_cache.shape with dim 1 (num_blocks) removed.
        self._per_block_shape = (
            len(kv_caches), first.shape[0], *first.shape[2:],
        )
        self._dtype = first.dtype
        # One block (all layers, both K and V) in bytes.
        single_block_elems = 1
        for d in self._per_block_shape:
            single_block_elems *= d
        self._per_block_bytes = single_block_elems * first.element_size()

        self._cpu_pool: CPUSwapPool | None = None
        if cpu_tier_capacity_bytes > 0:
            self._cpu_pool = CPUSwapPool(
                cpu_tier_capacity_bytes, self._per_block_bytes,
            )
            logger.info(
                "kv-swap: CPU tier enabled, capacity=%.1f MB "
                "(~%d blocks of %.1f MB each)",
                cpu_tier_capacity_bytes / (1024 ** 2),
                self._cpu_pool.max_blocks,
                self._per_block_bytes / (1024 ** 2),
            )
        else:
            logger.info(
                "kv-swap: CPU tier disabled, evictions write directly "
                "to %s (block size %.1f MB)",
                swap_dir,
                self._per_block_bytes / (1024 ** 2),
            )

    def submit_swap_outs(
        self,
        directives: list[tuple[str, int, int, int]],
    ) -> None:
        """Enqueue D2H + tier placement for each directive.

        ``directives`` entries are (req_id, group_idx, block_pos,
        gpu_block_id). The block id is the row index along dim 1 of
        each layer's KV cache tensor.
        """
        if not directives:
            return

        self._stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._stream):
            for req_id, group_idx, block_pos, block_id in directives:
                host = torch.empty(
                    self._per_block_shape,
                    dtype=self._dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for layer_idx, kv in enumerate(self._kv_caches):
                    host[layer_idx].copy_(
                        kv[:, block_id, ...], non_blocking=True
                    )
                event = torch.cuda.Event()
                event.record(self._stream)
                self._executor.submit(
                    self._wait_and_complete,
                    event, host, req_id, group_idx, block_pos,
                )

    def _wait_and_complete(
        self,
        event: torch.cuda.Event,
        host_tensor: torch.Tensor,
        req_id: str,
        group_idx: int,
        block_pos: int,
    ) -> None:
        """Block on D2H completion, then place in the appropriate tier
        and signal the swap-out as complete.

        With CPU tier enabled, completion fires as soon as bytes are
        in the pinned host buffer; demotion to NVMe (if the pool is
        over capacity) runs in the background and does not delay the
        ack. Without CPU tier, the file is written synchronously on
        this threadpool worker before completion fires.
        """
        try:
            event.synchronize()
        except Exception as e:
            logger.error(
                "kv swap-out D2H sync failed: %s/%d/%d (%s)",
                req_id, group_idx, block_pos, e,
            )
            return

        key = (req_id, group_idx, block_pos)

        if self._cpu_pool is not None:
            # L2 path: register in the CPU pool. If the pool was
            # already at capacity, demote the displaced entry to L3
            # in the background.
            evicted = self._cpu_pool.insert(key, host_tensor)
            if evicted is not None:
                old_key, old_tensor = evicted
                self._executor.submit(
                    self._demote_to_nvme, old_key, old_tensor,
                )
            # Ack fires immediately — the new entry is L2-resident.
            with self._lock:
                self._completed.append(
                    (req_id, group_idx, block_pos, "")
                )
            return

        # No CPU tier: write straight to L3 on this thread.
        file_path = self._nvme_path(req_id, group_idx, block_pos)
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Bytes-only on-disk format (paper §6.3): reinterpret as
            # uint8 to bypass numpy's missing BF16/FP8 dtype support.
            host_tensor.contiguous().view(torch.uint8).numpy().tofile(
                file_path
            )
        except Exception as e:
            logger.error(
                "kv swap-out write failed: %s (%s)", file_path, e,
            )
            return
        with self._lock:
            self._completed.append(
                (req_id, group_idx, block_pos, file_path)
            )

    def _demote_to_nvme(
        self,
        key: tuple[str, int, int],
        tensor: torch.Tensor,
    ) -> None:
        """Background L2 → L3 spill. Best-effort; failures are logged
        and the in-memory entry is dropped (recompute fallback covers
        a subsequent miss)."""
        req_id, group_idx, block_pos = key
        file_path = self._nvme_path(req_id, group_idx, block_pos)
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            tensor.contiguous().view(torch.uint8).numpy().tofile(file_path)
        except Exception as e:
            logger.error(
                "kv swap demotion failed: %s (%s)", file_path, e,
            )

    def _nvme_path(
        self, req_id: str, group_idx: int, block_pos: int
    ) -> str:
        return os.path.join(
            self._swap_dir, req_id, f"{group_idx}_{block_pos}.kv",
        )

    def collect_completed(
        self,
    ) -> list[tuple[str, int, int, str]] | None:
        """Drain swap-out completions accumulated since the last call."""
        with self._lock:
            if not self._completed:
                return None
            done = self._completed
            self._completed = []
        return done

    def collect_swap_in_results(
        self,
    ) -> tuple[
        list[tuple[str, int, int]] | None,
        list[tuple[str, int, int]] | None,
    ]:
        """Drain swap-in completions and failures since the last call.

        Returns (completed, failed). Either may be None if nothing
        accumulated; that signals "no work to report" and saves the
        scheduler from carrying empty lists in ModelRunnerOutput.
        """
        with self._lock:
            done = self._swap_in_completed or None
            failed = self._swap_in_failed or None
            if done is not None:
                self._swap_in_completed = []
            if failed is not None:
                self._swap_in_failed = []
        return done, failed

    def submit_swap_ins(
        self,
        directives: list[tuple[str, int, int, int]],
    ) -> None:
        """Restore each (req, group, pos) into the named fresh GPU block.

        Look-up order is L2 → L3 → fail. L2 hits issue an H2D copy
        directly on the swap stream. L3 hits dispatch to the threadpool
        (which reads the file into a pinned host buffer and then issues
        the H2D). Both paths emit the result via
        ``collect_swap_in_results`` once the H2D completes.
        """
        if not directives:
            return

        self._stream.wait_stream(torch.cuda.current_stream())

        for req_id, group_idx, block_pos, fresh_block_id in directives:
            key = (req_id, group_idx, block_pos)

            # L2 (CPU pool) lookup.
            host_tensor = None
            if self._cpu_pool is not None:
                host_tensor = self._cpu_pool.get(key)

            if host_tensor is not None:
                # Hot path: H2D from L2 immediately.
                self._issue_swap_in_h2d(host_tensor, fresh_block_id, key)
                continue

            # L3 (disk) lookup.
            file_path = self._nvme_path(req_id, group_idx, block_pos)
            if not os.path.exists(file_path):
                # Both tiers miss — bytes are lost. Emit failure.
                logger.error(
                    "kv swap-in: bytes not found for %s/%d/%d",
                    req_id, group_idx, block_pos,
                )
                with self._lock:
                    self._swap_in_failed.append(key)
                continue

            # Disk read on the threadpool — keeps the scheduler thread
            # off slow I/O.
            self._executor.submit(
                self._read_disk_and_swap_in,
                file_path, fresh_block_id, key,
            )

    def _issue_swap_in_h2d(
        self,
        host_tensor: torch.Tensor,
        fresh_block_id: int,
        key: tuple[str, int, int],
    ) -> None:
        """Schedule H2D on the swap stream from an in-memory pinned
        buffer, then dispatch a threadpool task to wait + ack."""
        with torch.cuda.stream(self._stream):
            for layer_idx, kv in enumerate(self._kv_caches):
                kv[:, fresh_block_id, ...].copy_(
                    host_tensor[layer_idx], non_blocking=True,
                )
        event = torch.cuda.Event()
        event.record(self._stream)
        self._executor.submit(self._wait_and_ack_swap_in, event, key)

    def _read_disk_and_swap_in(
        self,
        file_path: str,
        fresh_block_id: int,
        key: tuple[str, int, int],
    ) -> None:
        """Threadpool task: read L3 file into a transient pinned host
        buffer, then issue H2D and wait for the H2D to complete before
        emitting the ack."""
        try:
            host = self._read_pinned_buffer_from_disk(file_path)
        except Exception as e:
            logger.error(
                "kv swap-in disk read failed: %s (%s)", file_path, e,
            )
            with self._lock:
                self._swap_in_failed.append(key)
            return

        with torch.cuda.stream(self._stream):
            for layer_idx, kv in enumerate(self._kv_caches):
                kv[:, fresh_block_id, ...].copy_(
                    host[layer_idx], non_blocking=True,
                )
        event = torch.cuda.Event()
        event.record(self._stream)
        # Wait synchronously on this thread — scheduler thread is not
        # blocked. The pinned host buffer must outlive the H2D, which
        # this synchronization ensures before it falls out of scope.
        self._wait_and_ack_swap_in(event, key)

    def _read_pinned_buffer_from_disk(
        self, file_path: str,
    ) -> torch.Tensor:
        """Read raw block bytes from disk into a fresh pinned host
        buffer. Symmetric to the swap-out write — bytes-only format,
        dtype-agnostic, works for BF16/FP8/FP16/FP32."""
        import numpy as np
        host = torch.empty(
            self._per_block_shape,
            dtype=self._dtype,
            device="cpu",
            pin_memory=True,
        )
        arr = np.fromfile(file_path, dtype=np.uint8)
        expected_bytes = host.numel() * host.element_size()
        if arr.nbytes != expected_bytes:
            raise IOError(
                f"swap-in size mismatch at {file_path}: "
                f"got {arr.nbytes} bytes, expected {expected_bytes}"
            )
        host_u8 = host.view(torch.uint8)
        host_u8.copy_(torch.from_numpy(arr).reshape(host_u8.shape))
        return host

    def _wait_and_ack_swap_in(
        self,
        event: torch.cuda.Event,
        key: tuple[str, int, int],
    ) -> None:
        """Wait for the H2D copy to complete; emit completion or
        failure. On success, also drop the L2 entry (the bytes are
        back on the GPU and the L2 copy is redundant)."""
        try:
            event.synchronize()
        except Exception as e:
            logger.error("kv swap-in H2D sync failed: %s (%s)", key, e)
            with self._lock:
                self._swap_in_failed.append(key)
            return
        if self._cpu_pool is not None:
            self._cpu_pool.remove(key)
        with self._lock:
            self._swap_in_completed.append(key)

    def drop_request_state(self, req_id: str) -> None:
        """Release CPU-tier entries for a finished request.

        Files on disk are not unlinked here — the scheduler-side
        ``SwapStore`` already tracks paths and unlinks them via its own
        finish path. (Files written by background L2→L3 demotion
        *after* the scheduler dropped the request can be left orphaned;
        they're swept on process restart since ``swap_dir`` is scratch.)
        """
        if self._cpu_pool is not None:
            self._cpu_pool.drop_request(req_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
