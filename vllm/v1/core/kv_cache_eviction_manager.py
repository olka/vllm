# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache page eviction manager.

Coordinates per-request page eviction. The Scheduler delegates four
decisions here:

  - before_schedule: proactive sweep + page-fault rollback
  - try_free_blocks: reactive eviction in place of preemption
  - absorb_output: ingest worker importance scores (EMA, heatmap)
  - on_request_finish: drop per-request eviction state

All policy lives here (alpha=0.3 EMA, max_eviction_fraction, head/tail
protection, heatmap cadence). The scheduler holds only a reference to
the manager and four hook call sites.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.block_importance import EvictedBlockInfo, QuestBlockStats
from vllm.v1.core.kv_cache_eviction_strategy import EvictionStrategy
from vllm.v1.core.kv_cache_swap import SwapStore
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.config.kv_cache_eviction import KVCacheEvictionConfig
    from vllm.v1.core.block_pool import KVCacheBlock
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


# Smoothing weight for EMA over per-step block importance scores.
_EMA_ALPHA = 0.3
# Heatmap is rewritten every Nth scoring cycle (HTML auto-refreshes every 5s).
_HEATMAP_WRITE_EVERY = 2
# Worker computes block importance every Nth step.
_BI_REQUEST_EVERY = 50
# Pre-warm fraction below the proactive threshold at which we start
# requesting block importance from the worker.
_BI_REQUEST_FILL_RATIO = 0.9
# Periodic state log cadence.
_STATE_LOG_EVERY = 100


class KVCacheEvictionManager:
    """Owns all page eviction state and policy for one Scheduler.

    The manager holds *references* to the scheduler's running list,
    requests dict, model name, and KVCacheManager — it does not own them,
    only reads from them.
    """

    def __init__(
        self,
        config: "KVCacheEvictionConfig",
        kv_cache_manager: "KVCacheManager",
        running: list["Request"],
        requests: dict[str, "Request"],
        max_model_len: int,
        block_size: int,
        model_name: str,
        strategy: EvictionStrategy | None = None,
    ) -> None:
        self._config = config
        self._kv_cache_manager = kv_cache_manager
        self._running = running
        self._requests = requests
        self._max_model_len = max_model_len
        self._block_size = block_size
        self._model_name = model_name
        self._strategy = strategy or EvictionStrategy.create(
            config.scoring_strategy
        )

        # Cached per-request block importance scores (EMA-smoothed).
        # Maps req_id -> dict(block_position -> score).
        self._score_cache: dict[str, dict[int, float]] = {}
        # Persistent cooldown for requests that were rolled back.
        # Cleared only on request finish (not each step). Prevents the
        # evict→rollback→recompute→evict thrash loop.
        self._cooldown_req_ids: set[str] = set()
        # Edge-triggered re-arming for proactive eviction: per-request
        # ``num_computed_tokens`` at the last successful eviction. The
        # gate in ``before_schedule`` requires
        # ``num_computed_tokens - last >= threshold * max_model_len``
        # before firing again, so eviction fires once per threshold's
        # worth of context growth (e.g., at 20%, 40%, 60%, ...) rather
        # than every cycle while ``context_fill > threshold`` holds.
        # Updated only when ``_evict_from_request`` returned >0 — silent
        # no-op sweeps (Otsu found nothing dead, evictable list empty)
        # don't push the gate forward.
        self._tokens_at_last_eviction: dict[str, int] = {}

        self._heatmap_counter = 0
        self._bi_step_counter = 0
        self._proactive_log_counter = 0
        self._first_eviction_dumped = False
        # Dynamic dead-block threshold. Recomputed each scoring cycle in
        # absorb_output: either via Otsu over the live score histogram
        # (if config.enable_otsu_threshold) or held at the strategy's
        # static fallback. Initialized to the static fallback so any
        # eviction call before the first scoring cycle has a value.
        self._dead_threshold: float = getattr(
            self._strategy, "_DEAD_THRESHOLD", 1e-4
        )

        # Swap-out state. None when ``config.enable_swap`` is False —
        # eviction then takes the legacy "drop and recompute" path.
        self._swap_store: SwapStore | None = None
        # (req_id, group_idx, block_pos) -> KVCacheBlock kept allocated
        # while the worker copies its KV bytes to disk. Freed once
        # ``absorb_output`` ingests the matching swap_out_completed ack.
        self._pending_swap_blocks: dict[
            tuple[str, int, int], "KVCacheBlock"
        ] = {}
        # Buffered directives drained by the scheduler each step into
        # SchedulerOutput.swap_out_blocks.
        self._pending_swap_directives: list[
            tuple[str, int, int, int]
        ] = []
        # Swap-IN state. Mirrors the swap-OUT pattern but in reverse:
        # the scheduler pre-allocates a fresh GPU block, queues a
        # directive for the worker to fill it from the swap tier, and
        # waits for ack before patching the request's block table.
        # Map keyed by (req_id, group_idx, block_pos) -> fresh block
        # holding the slot until ack.
        self._pending_swap_in_blocks: dict[
            tuple[str, int, int], "KVCacheBlock"
        ] = {}
        # Buffered directives drained into SchedulerOutput.swap_in_blocks.
        self._pending_swap_in_directives: list[
            tuple[str, int, int, int]
        ] = []
        # Transient set of requests with at least one in-flight
        # swap-in. Used to gate proactive eviction (don't evict more
        # from a request that's still recovering bytes). Distinct from
        # the persistent ``_cooldown_req_ids`` — a request that
        # successfully swaps in must remain eligible for future
        # eviction, otherwise swap defeats its own purpose.
        self._pending_swap_in_req_ids: set[str] = set()
        # Background thread for fire-and-forget file unlinks on request
        # finish. Created lazily.
        self._swap_unlink_pool: ThreadPoolExecutor | None = None
        if config.enable_swap:
            self._swap_store = SwapStore()
            os.makedirs(config.swap_dir, exist_ok=True)
            self._swap_unlink_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kv-swap-unlink"
            )

        logger.error(
            "KV cache eviction config loaded: enable=%s, "
            "scoring_strategy=%s, proactive_threshold=%.2f",
            self._config.enable,
            self._config.scoring_strategy,
            self._config.proactive_eviction_threshold,
        )

    # ─────────────────────────────────────────────────────────────────
    # Public API — four hook methods.
    # ─────────────────────────────────────────────────────────────────

    def before_schedule(self) -> set[str]:
        """Run proactive sweep + page-fault rollback.

        Returns the set of request IDs that were rolled back this step.
        The caller (Scheduler) MUST assign this set to its own
        `_rolled_back_req_ids` field — multiple downstream readers in
        scheduler.py consume it off `self`.
        """
        self._proactive_log_counter += 1

        threshold = self._config.proactive_eviction_threshold
        for request in list(self._running):
            context_fill = request.num_computed_tokens / self._max_model_len
            # Edge-triggered re-arm: require at least one threshold's
            # worth of context growth since the last successful eviction
            # for this request. Without this, eviction fires every
            # cycle while context_fill > threshold, which produces a
            # marginal 1-block-per-step churn as scores drift across
            # the dead boundary. With it, eviction fires at multiples
            # of the threshold (20%, 40%, 60%, ...) — once per
            # meaningful chunk of new context.
            last_evict_tokens = self._tokens_at_last_eviction.get(
                request.request_id, 0
            )
            tokens_since_evict = (
                request.num_computed_tokens - last_evict_tokens
            )
            growth_since_evict = tokens_since_evict / self._max_model_len
            re_arm_tokens_needed = int(threshold * self._max_model_len)
            re_arm_ready = growth_since_evict >= threshold
            if self._proactive_log_counter % _STATE_LOG_EVERY == 1:
                if re_arm_ready:
                    re_arm_str = "ready"
                else:
                    re_arm_str = (
                        f"{tokens_since_evict}/{re_arm_tokens_needed}"
                        f" tokens since last evict"
                    )
                logger.error(
                    "Eviction check step=%d: req %s "
                    "context_fill=%.1f%% (%d/%d tokens), "
                    "threshold=%.1f%%, cooldown=%s, "
                    "dead_thr=%.6e (otsu=%s), re_arm=%s",
                    self._proactive_log_counter,
                    request.request_id,
                    context_fill * 100,
                    request.num_computed_tokens,
                    self._max_model_len,
                    threshold * 100,
                    request.request_id in self._cooldown_req_ids,
                    self._dead_threshold,
                    "on" if self._config.enable_otsu_threshold else "off",
                    re_arm_str,
                )
            if (context_fill > threshold
                    and re_arm_ready
                    and request.request_id not in self._cooldown_req_ids
                    and request.request_id
                    not in self._pending_swap_in_req_ids):
                n_evicted = self._evict_from_request(request)
                if n_evicted > 0:
                    self._tokens_at_last_eviction[request.request_id] = (
                        request.num_computed_tokens
                    )

        return self._handle_page_faults()

    def try_free_blocks(
        self, num_needed: int, exclude: "Request"
    ) -> bool:
        """Reactive eviction: try to free blocks before preemption.

        Returns True if at least one block was freed and the caller
        should retry allocation; False if the scheduler should fall
        through to full preemption.
        """
        return self._evict_from_victim(num_needed, exclude) > 0

    def take_swap_directives(self) -> list[tuple[str, int, int, int]] | None:
        """Drain buffered swap-out directives for the next SchedulerOutput.

        Each entry is (req_id, kv_cache_group_idx, block_position,
        gpu_block_id). Returns None when there is nothing pending — the
        scheduler treats None and empty list identically; None just
        avoids serializing an empty list every step.
        """
        if not self._pending_swap_directives:
            return None
        directives = self._pending_swap_directives
        self._pending_swap_directives = []
        return directives

    def take_swap_in_directives(
        self,
    ) -> list[tuple[str, int, int, int]] | None:
        """Drain buffered swap-in directives for the next SchedulerOutput.

        Each entry is (req_id, kv_cache_group_idx, block_position,
        fresh_gpu_block_id) — the fresh block has been allocated by
        ``_try_swap_in`` and is awaiting the worker to populate it.
        """
        if not self._pending_swap_in_directives:
            return None
        directives = self._pending_swap_in_directives
        self._pending_swap_in_directives = []
        return directives

    def absorb_output(self, output: "ModelRunnerOutput") -> None:
        """Ingest worker importance scores; apply EMA; periodic heatmap.

        Also processes swap-out acks (free pinned blocks, record paths
        in SwapStore) and swap-in acks/failures (patch block table on
        success, recover state on failure).
        """
        self._absorb_swap_out_acks(output)
        self._absorb_swap_in_acks(output)
        self._absorb_swap_in_failures(output)

        if not output.block_importance_scores:
            return

        for req_id, scores in output.block_importance_scores.items():
            existing = self._score_cache.get(req_id)
            if existing is None:
                self._score_cache[req_id] = {pos: s for pos, s in scores}
            else:
                for pos, s in scores:
                    if pos in existing:
                        existing[pos] = (
                            (1 - _EMA_ALPHA) * existing[pos]
                            + _EMA_ALPHA * s
                        )
                    else:
                        existing[pos] = s

        # Refresh the dead-block threshold for this cycle. Otsu when
        # enabled and we have data; static fallback otherwise.
        self._dead_threshold = self._compute_dead_threshold()

        self._heatmap_counter += 1
        if self._heatmap_counter % _HEATMAP_WRITE_EVERY == 0:
            self._maybe_write_heatmap()

    def on_request_finish(self, req_id: str) -> None:
        """Drop per-request eviction state (called from finish path)."""
        self._score_cache.pop(req_id, None)
        self._cooldown_req_ids.discard(req_id)
        self._tokens_at_last_eviction.pop(req_id, None)
        self._drop_swap_state(req_id)

    def _absorb_swap_out_acks(self, output: "ModelRunnerOutput") -> None:
        """Free pinned blocks for completed swap-outs, record paths."""
        if self._swap_store is None or not output.swap_out_completed:
            return
        for req_id, group_idx, block_pos, path in output.swap_out_completed:
            entry = self._swap_store.complete(req_id, block_pos, path)
            if entry is None:
                # Request finished between directive and ack — the
                # block was already released; the file (if any) will
                # be unlinked by drop_swap_state. Best-effort cleanup
                # here as well in case the request is gone.
                self._unlink_path_async(path)
                continue
            block = self._pending_swap_blocks.pop(
                (req_id, group_idx, block_pos), None
            )
            if block is not None:
                self._kv_cache_manager.block_pool.free_blocks([block])

    def _absorb_swap_in_acks(self, output: "ModelRunnerOutput") -> None:
        """Patch block table for blocks the worker successfully restored.

        For each successful swap-in:
          1. Pop the pre-allocated fresh GPU block from
             ``_pending_swap_in_blocks``.
          2. Call ``kv_cache_manager.attach_block_at_position`` to
             replace the null_block at ``block_pos`` with the fresh,
             now-populated block.
          3. Drop the SwapStore entry (bytes are back on the GPU; the
             on-disk file or L2 buffer is no longer needed).
          4. Drop the per-position eviction record so the next page
             fault cycle does not retry recovery for this position.
        """
        if self._swap_store is None or not output.swap_in_completed:
            return
        for req_id, group_idx, block_pos in output.swap_in_completed:
            key = (req_id, group_idx, block_pos)
            new_block = self._pending_swap_in_blocks.pop(key, None)
            if new_block is None:
                # No pre-allocated block — request was likely finished
                # between directive and ack. Drop the swap entry and
                # move on. The bytes-on-GPU are already in a freshly
                # allocated block_id; that block belongs to nobody now
                # and will be reaped via the existing block-pool path
                # when the worker stops referencing it.
                self._swap_store.consume(req_id, block_pos)
                self._release_swap_in_gate(req_id)
                continue
            try:
                self._kv_cache_manager.attach_block_at_position(
                    req_id, group_idx, block_pos, new_block,
                )
            except (IndexError, ValueError) as e:
                # Block table no longer matches expectations (request
                # finished, or the position is no longer null). Free
                # the fresh block; drop the swap entry. Conservative
                # recovery — log loudly so we notice if this fires.
                logger.error(
                    "swap-in ack: failed to attach req=%s pos=%d: %s",
                    req_id, block_pos, e,
                )
                self._kv_cache_manager.block_pool.free_blocks([new_block])
                self._swap_store.consume(req_id, block_pos)
                self._release_swap_in_gate(req_id)
                continue
            # Bytes are back on the GPU. Drop swap entry + eviction
            # record so subsequent page-fault cycles ignore this
            # position.
            self._swap_store.consume(req_id, block_pos)
            self._kv_cache_manager.unrecord_evicted(req_id, block_pos)
            self._release_swap_in_gate(req_id)
            logger.error(
                "Page fault swap-in completed: req %s, pos %d "
                "(block table patched, no rollback)",
                req_id, block_pos,
            )

    def _absorb_swap_in_failures(
        self, output: "ModelRunnerOutput"
    ) -> None:
        """Recover state for blocks the worker could not restore.

        The pre-allocated fresh GPU block is freed back to the pool;
        the SwapStore entry is dropped (bytes are lost). The per-
        position eviction record is *kept* — the next page fault
        cycle will attempt recovery again, find no swap entry, and
        fall through to rollback. Self-healing.
        """
        if self._swap_store is None or not output.swap_in_failed:
            return
        for req_id, group_idx, block_pos in output.swap_in_failed:
            key = (req_id, group_idx, block_pos)
            new_block = self._pending_swap_in_blocks.pop(key, None)
            if new_block is not None:
                self._kv_cache_manager.block_pool.free_blocks([new_block])
            # Drop the (now invalid) swap entry; next cycle rolls back.
            self._swap_store.consume(req_id, block_pos)
            self._release_swap_in_gate(req_id)
            logger.warning(
                "swap-in failed: req=%s group=%d pos=%d — bytes lost; "
                "next page-fault cycle will rollback",
                req_id, group_idx, block_pos,
            )

    def _drop_swap_state(self, req_id: str) -> None:
        """Release any swap state held for a finished request.

        Pending swap-out blocks: freed (worker may or may not have
        completed the copy — either way the request is gone).
        Pending swap-in blocks: freed (the fresh allocation is no
        longer needed; the worker's H2D copy, if it lands, will write
        into a block that nobody references).
        Completed swap-out paths: unlinked asynchronously.
        """
        if self._swap_store is None:
            return
        pending, paths = self._swap_store.drop_request(req_id)
        # Free any pending swap-out blocks.
        if pending:
            stale_swap_out: list["KVCacheBlock"] = []
            for key in list(self._pending_swap_blocks.keys()):
                if key[0] == req_id:
                    stale_swap_out.append(
                        self._pending_swap_blocks.pop(key)
                    )
            if stale_swap_out:
                self._kv_cache_manager.block_pool.free_blocks(
                    stale_swap_out
                )
        # Free any pending swap-in blocks (independent of pending
        # swap-out — a request can have both at different positions).
        stale_swap_in: list["KVCacheBlock"] = []
        for key in list(self._pending_swap_in_blocks.keys()):
            if key[0] == req_id:
                stale_swap_in.append(
                    self._pending_swap_in_blocks.pop(key)
                )
        if stale_swap_in:
            self._kv_cache_manager.block_pool.free_blocks(stale_swap_in)
        # Always discard from the gate — request is gone, no swap-ins
        # to wait on.
        self._pending_swap_in_req_ids.discard(req_id)
        for path in paths:
            self._unlink_path_async(path)

    def _unlink_path_async(self, path: str) -> None:
        if self._swap_unlink_pool is None:
            return
        self._swap_unlink_pool.submit(_safe_unlink, path)

    def _compute_dead_threshold(self) -> float:
        """Pick the dead-block threshold for the current scoring cycle.

        With Otsu disabled (default), returns the strategy's static
        ``_DEAD_THRESHOLD`` ClassVar.

        With Otsu enabled, computes the threshold via Otsu's method
        over the union of all per-request EMA scores. Falls back to
        the static threshold when the score cache is empty (warm-up).
        """
        static_fallback = getattr(
            self._strategy, "_DEAD_THRESHOLD", 1e-4
        )
        if not self._config.enable_otsu_threshold:
            return static_fallback

        all_scores: list[float] = []
        for per_req in self._score_cache.values():
            all_scores.extend(per_req.values())
        if not all_scores:
            return static_fallback

        from vllm.v1.core.otsu import otsu_threshold

        thr = otsu_threshold(all_scores)
        # Sanity guard: if Otsu picks something pathological (e.g.,
        # below the static fallback by orders of magnitude), prefer
        # the fallback. The reverse — Otsu above the fallback — is
        # the expected behavior at short traces and we trust it.
        return max(thr, 0.0)

    def should_request_block_importance(
        self, scheduled_running: list["Request"]
    ) -> bool:
        """Whether the worker should compute block importance this step.

        Returns False unconditionally for strategies that don't need
        worker scoring. Otherwise gated by step counter + per-request
        fill ratio.
        """
        if not self._strategy.needs_worker_scoring:
            return False
        self._bi_step_counter += 1
        if self._bi_step_counter % _BI_REQUEST_EVERY != 0:
            return False
        threshold = (
            self._config.proactive_eviction_threshold * _BI_REQUEST_FILL_RATIO
        )
        return any(
            r.num_computed_tokens / self._max_model_len > threshold
            for r in scheduled_running
        )

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    # ─────────────────────────────────────────────────────────────────
    # Private — moved verbatim from Scheduler.
    # ─────────────────────────────────────────────────────────────────

    def _handle_page_faults(self) -> set[str]:
        """Recover evicted blocks under memory pressure.

        For each request with evicted blocks, try (in order):

          1. **Swap-in** the earliest evicted position (Option B —
             one fault at a time). If the SwapStore has the entry and
             a fresh GPU block is available, a directive is queued
             for the worker; the request stalls one step waiting for
             ack, after which ``_absorb_swap_in_acks`` patches the
             block table.
          2. **Rollback** to the earliest evicted position. Falls
             through here when swap is disabled, when no swap entry
             exists for this request, or when fresh-block allocation
             failed.

        Recovery only fires within the rollback band:
            min_usage ≤ block_pool_usage ≤ max_usage

        Below `min_usage`: no real memory pressure → leave evictions in
        place (proactive eviction was for per-request context fill, not
        GPU pressure; undoing it serves no purpose).
        Above `max_usage`: too tight to afford recomputation or fresh
        allocation → leave evictions in place to avoid memory blowup.

        Both swap-in and rollback add the request to the returned
        ``rolled_back`` set — the scheduler skips the request this
        step in both cases. The semantic difference is what happens
        on resume: rollback recomputes from earliest_pos via chunked
        prefill, while swap-in resumes from the request's current
        ``num_computed_tokens`` once the bytes are back on the GPU.
        """
        rolled_back: set[str] = set()
        evicted_map = self._kv_cache_manager.evicted_blocks
        if not evicted_map:
            return rolled_back

        # TODO/FIXME: proactive recovery is gated off by default. The
        # body below has no demand signal — it picks the earliest
        # evicted position from ``evicted_blocks`` every cycle and
        # eagerly tries to recover it (swap-in or rollback). Combined
        # with ``null_block`` zeroing evicted positions (which makes
        # demand unobservable to the model), this thrashes the lowest
        # cold position — evict, recover, evict, recover — instead of
        # producing the paper's "cold blocks stay cold" dynamics.
        # Re-enable only after wiring real demand-paged recovery (see
        # KVCacheEvictionConfig.enable_proactive_recovery docstring).
        # Reactive recovery via ``try_free_blocks`` (the near-preemption
        # path) is independent of this gate and remains active.
        if not self._config.enable_proactive_recovery:
            return rolled_back

        usage = self._kv_cache_manager.usage
        if (usage < self._config.page_fault_min_usage_threshold
                or usage > self._config.page_fault_usage_threshold):
            return rolled_back

        for request_id in list(evicted_map.keys()):
            request = self._requests.get(request_id)
            if request is None or request.status != RequestStatus.RUNNING:
                evicted_map.pop(request_id, None)
                continue

            evicted_blocks = evicted_map[request_id]
            if not evicted_blocks:
                evicted_map.pop(request_id, None)
                continue

            earliest_pos = min(
                info.quest_stats.block_position
                for info in evicted_blocks
            )

            # Skip if a swap-in for this position is already in flight
            # (request stalled, awaiting ack). Don't issue a duplicate.
            if (request_id, 0, earliest_pos) in self._pending_swap_in_blocks:
                rolled_back.add(request_id)
                continue

            # Skip if swap-OUT for this position is in flight — the
            # directive was queued in the same scheduler tick (proactive
            # eviction sweep above) and the worker ack hasn't landed yet
            # so SwapStore._completed is empty. Stall this cycle without
            # rollback or cooldown; the next cycle will find the ack in
            # SwapStore._completed and _try_swap_in will succeed.
            if (self._swap_store is not None
                    and self._swap_store.is_pending(
                        request_id, earliest_pos)):
                rolled_back.add(request_id)
                continue

            # Try swap-in first; on success, the request stalls but the
            # evicted_map entry is preserved (consumed on ack) and
            # num_computed_tokens is left untouched. Do NOT add to
            # ``_cooldown_req_ids`` — that set is persistent and would
            # permanently disable proactive eviction for this request,
            # defeating the point of swap. Gating is handled by the
            # transient ``_pending_swap_in_req_ids`` set instead, which
            # auto-clears once the worker acks (or the swap fails).
            if self._try_swap_in(request_id, earliest_pos):
                if request.spec_token_ids:
                    request.spec_token_ids = []
                rolled_back.add(request_id)
                logger.error(
                    "Page fault swap-in queued: req %s, pos %d, "
                    "computed_tokens %d (unchanged)",
                    request_id, earliest_pos,
                    request.num_computed_tokens,
                )
                continue

            # Fall-through: rollback path. Same as legacy behavior.
            self._kv_cache_manager.rollback_blocks_from_position(
                request_id, earliest_pos
            )

            new_computed = earliest_pos * self._block_size
            old_computed = request.num_computed_tokens
            request.num_computed_tokens = min(
                new_computed, request.num_computed_tokens
            )
            logger.error(
                "Page fault rollback: req %s, %d -> %d computed tokens",
                request_id, old_computed, request.num_computed_tokens,
            )

            if request.spec_token_ids:
                request.spec_token_ids = []

            rolled_back.add(request_id)
            self._cooldown_req_ids.add(request_id)
            evicted_map.pop(request_id, None)

        return rolled_back

    def _try_swap_in(self, request_id: str, block_pos: int) -> bool:
        """Try to queue a swap-in directive for a single evicted block.

        Returns True iff the directive was queued — caller stalls the
        request and awaits the worker's ack. Returns False when:
          - swap is disabled (no SwapStore),
          - no swap entry exists for (request_id, block_pos), or
          - fresh GPU block allocation fails (memory pressure).

        On False the caller must fall through to rollback.
        """
        if self._swap_store is None:
            return False
        if self._swap_store.lookup(request_id, block_pos) is None:
            return False
        try:
            new_blocks = (
                self._kv_cache_manager.block_pool.get_new_blocks(1)
            )
        except ValueError:
            # No free GPU block — we cannot pre-allocate the swap-in
            # destination. The caller will rollback, which doesn't need
            # a fresh allocation.
            return False
        new_block = new_blocks[0]
        # Group index 0 — single-group assumption (consistent with
        # swap-out; multi-group hybrid models are not supported).
        group_idx = 0
        key = (request_id, group_idx, block_pos)
        self._pending_swap_in_blocks[key] = new_block
        self._pending_swap_in_req_ids.add(request_id)
        self._pending_swap_in_directives.append(
            (request_id, group_idx, block_pos, new_block.block_id)
        )
        return True

    def _release_swap_in_gate(self, req_id: str) -> None:
        """Discard ``req_id`` from ``_pending_swap_in_req_ids`` once
        no in-flight swap-ins remain for that request. Re-enables
        proactive eviction the next ``before_schedule`` cycle."""
        if any(k[0] == req_id for k in self._pending_swap_in_blocks):
            return
        self._pending_swap_in_req_ids.discard(req_id)

    def _evict_from_victim(
        self, num_needed: int, exclude: "Request"
    ) -> int:
        """Reactive: pick victim with most blocks, evict its coldest.

        Reactive eviction is the panic path: allocation has already
        failed and the scheduler needs blocks back this step. Swap-out
        is async (worker-side D2H + disk write), so this path always
        takes the legacy drop-and-recompute route even when
        ``enable_swap`` is on — waiting for an async write would defeat
        the purpose of reactive eviction. Proactive eviction is where
        swap pays off.
        """
        victim = self._select_victim(exclude)
        if victim is None:
            return 0

        all_blocks = self._kv_cache_manager.coordinator.get_blocks(
            victim.request_id
        )
        if not all_blocks:
            return 0
        blocks = max(all_blocks, key=len)

        total_blocks = len(blocks)
        cfg = self._config
        num_protected_head = min(cfg.num_protected_head_blocks, total_blocks)
        num_protected_tail = min(
            cfg.num_protected_tail_blocks,
            total_blocks - num_protected_head,
        )
        tail_start = total_blocks - num_protected_tail

        evictable = [
            (i, b) for i, b in enumerate(blocks)
            if not b.is_null
            and i >= num_protected_head
            and i < tail_start
        ]
        if not evictable:
            return 0

        importance = self._score_cache.get(victim.request_id)
        evictable.sort(
            key=lambda ib: self._strategy.score_block(
                ib[0], ib[1],
                importance.get(ib[0]) if importance is not None else None,
            )
        )

        max_evict = max(1, int(total_blocks * cfg.max_eviction_fraction))
        num_to_evict = min(num_needed, max_evict, len(evictable))
        to_evict = evictable[:num_to_evict]
        positions_to_evict = [pos for pos, _ in to_evict]

        block_details = []
        for pos, blk in to_evict:
            imp_score = (importance.get(pos, None)
                         if importance is not None else None)
            block_details.append(
                f"pos={pos} access_ts={blk.last_accessed}"
                f" importance={imp_score}"
            )
        logger.error(
            "Page eviction (reactive): req %s, %d/%d blocks evicted "
            "(strategy=%s) blocks=[%s]",
            victim.request_id,
            len(positions_to_evict),
            total_blocks,
            self._strategy.name,
            ", ".join(block_details),
        )

        self._kv_cache_manager.evict_blocks_at_positions(
            victim.request_id, positions_to_evict
        )
        self._record_evicted(victim.request_id, positions_to_evict)

        if victim.spec_token_ids:
            victim.spec_token_ids = []

        return len(positions_to_evict)

    def _select_victim(self, exclude: "Request") -> "Request | None":
        """Pick the running request with the most allocated blocks."""
        best: "Request | None" = None
        best_blocks = 0
        for req in self._running:
            if req is exclude:
                continue
            num_blocks = (
                self._kv_cache_manager.get_num_blocks_for_request(
                    req.request_id
                )
            )
            if num_blocks > best_blocks:
                best_blocks = num_blocks
                best = req
        if best_blocks < self._config.min_blocks_for_eviction:
            return None
        return best

    def _evict_from_request(self, request: "Request") -> int:
        """Proactive: evict dead blocks (EMA score below epsilon)."""
        all_blocks = self._kv_cache_manager.coordinator.get_blocks(
            request.request_id
        )
        if not all_blocks:
            logger.error(
                "evict_skip req=%s reason=no_blocks", request.request_id)
            return 0

        blocks = max(all_blocks, key=len)
        total_blocks = len(blocks)
        cfg = self._config
        if total_blocks < cfg.min_blocks_for_eviction:
            logger.error(
                "evict_skip req=%s reason=too_few_blocks total=%d min=%d",
                request.request_id, total_blocks,
                cfg.min_blocks_for_eviction)
            return 0

        num_protected_head = min(cfg.num_protected_head_blocks, total_blocks)
        num_protected_tail = min(
            cfg.num_protected_tail_blocks,
            total_blocks - num_protected_head,
        )
        tail_start = total_blocks - num_protected_tail

        evictable = [
            (i, b) for i, b in enumerate(blocks)
            if not b.is_null
            and i >= num_protected_head
            and i < tail_start
        ]
        if not evictable:
            # Post-aggressive-eviction steady state — everything outside
            # the head/tail protection has already been evicted to
            # null_block. Silent on purpose; fires every sweep at
            # tight budgets and pollutes the log.
            return 0

        # Proactive eviction is strategy-defined: a block is evicted only
        # if strategy.is_dead(score) returns True. Strategies that don't
        # opt into proactive eviction (default is_dead → False) bail here.
        importance = self._score_cache.get(request.request_id)
        if self._strategy.needs_worker_scoring and importance is None:
            logger.error(
                "evict_skip req=%s reason=no_scores_yet "
                "(strategy=%s, _bi_step_counter=%d)",
                request.request_id, self._strategy.name,
                self._bi_step_counter)
            return 0

        def _score(ib: tuple[int, "KVCacheBlock"]) -> float:
            return self._strategy.score_block(
                ib[0], ib[1],
                importance.get(ib[0]) if importance is not None else None,
            )

        evictable.sort(key=_score)

        max_evict = max(1, int(total_blocks * cfg.max_eviction_fraction))

        dead_thr = self._dead_threshold
        dead_blocks = [
            (pos, blk) for pos, blk in evictable
            if self._strategy.is_dead(_score((pos, blk)), dead_thr)
        ]
        if not dead_blocks:
            # Post-eviction steady state — Otsu has tightened past the
            # residual distribution. Silent on purpose; this fires every
            # sweep and pollutes the log otherwise. Re-enable for
            # diagnostics by uncommenting the logger.error below.
            return 0
        num_to_evict = min(max_evict, len(dead_blocks))
        to_evict = dead_blocks[:num_to_evict]
        positions_to_evict = [pos for pos, _ in to_evict]

        block_details = []
        for pos, _ in to_evict:
            imp_score = (
                importance.get(pos, None) if importance is not None else None
            )
            block_details.append(
                f"pos={pos} score={imp_score:.6f}"
                if imp_score is not None
                else f"pos={pos} score=None"
            )
        score_summary = ""
        if importance:
            all_scores = list(importance.values())
            score_summary = (
                f" score_range=[{min(all_scores):.6f}"
                f"..{max(all_scores):.6f}]"
                f" ratio={max(all_scores)/max(min(all_scores),1e-10):.0f}x"
            )
        logger.error(
            "Page eviction (proactive, %s): req %s, %d/%d blocks evicted%s "
            "blocks=[%s]",
            self._strategy.name,
            request.request_id,
            len(positions_to_evict),
            total_blocks,
            score_summary,
            ", ".join(block_details),
        )

        first_dump = not self._first_eviction_dumped
        if first_dump:
            logger.error(
                "First-eviction snapshot (BEFORE): req=%s, "
                "%d blocks about to evict",
                request.request_id, len(positions_to_evict),
            )
            self._maybe_write_heatmap()

        if self._swap_store is None:
            self._kv_cache_manager.evict_blocks_at_positions(
                request.request_id, positions_to_evict
            )
        else:
            # Swap path: detach without freeing; pin the (pos, block)
            # pairs we already picked and queue one directive each for
            # the worker. Single-group only — hybrid (e.g. Mamba+Attn)
            # KV layouts under swap are not supported yet.
            self._kv_cache_manager.evict_blocks_at_positions(
                request.request_id,
                positions_to_evict,
                free_immediately=False,
            )
            group_idx = 0
            req_id = request.request_id
            for pos, blk in to_evict:
                self._swap_store.pin(req_id, pos, group_idx, blk.block_id)
                self._pending_swap_blocks[(req_id, group_idx, pos)] = blk
                self._pending_swap_directives.append(
                    (req_id, group_idx, pos, blk.block_id)
                )
        self._record_evicted(request.request_id, positions_to_evict)

        if first_dump:
            logger.error(
                "First-eviction snapshot (AFTER): req=%s, "
                "%d blocks evicted",
                request.request_id, len(positions_to_evict),
            )
            self._maybe_write_heatmap()
            self._first_eviction_dumped = True

        if request.spec_token_ids:
            request.spec_token_ids = []

        return len(positions_to_evict)

    def _record_evicted(
        self, req_id: str, positions: list[int]
    ) -> None:
        """Record placeholder Quest stats for evicted blocks.

        Real Quest stats (key_min/max) require GPU access on the worker
        side. Until that lands, eviction records placeholders so that
        page-fault recovery still has a position map to roll back from.
        """
        evicted_list = self._kv_cache_manager.evicted_blocks.setdefault(
            req_id, []
        )
        bs = self._block_size
        for pos in positions:
            evicted_list.append(EvictedBlockInfo(
                quest_stats=QuestBlockStats(
                    key_min=None,  # type: ignore[arg-type]
                    key_max=None,  # type: ignore[arg-type]
                    token_ids=[],
                    position_start=pos * bs,
                    position_end=(pos + 1) * bs,
                    block_position=pos,
                ),
                kv_cache_group_id=0,
            ))

    def _maybe_write_heatmap(self) -> None:
        """Write KV cache attention heatmap HTML."""
        if not self._score_cache:
            return

        from vllm.v1.core.kv_cache_heatmap import write_heatmap

        cfg = self._config
        head_n = cfg.num_protected_head_blocks
        tail_n = cfg.num_protected_tail_blocks

        requests_data = []
        for req_id, scores in self._score_cache.items():
            req = self._requests.get(req_id)
            if req is None:
                continue
            all_blocks = self._kv_cache_manager.coordinator.get_blocks(
                req_id
            )
            if not all_blocks:
                continue
            total = len(max(all_blocks, key=len))

            evicted = set()
            evicted_meta = (
                self._kv_cache_manager.evicted_blocks.get(req_id, [])
            )
            for info in evicted_meta:
                evicted.add(info.quest_stats.block_position)

            block_token_ids: dict[int, list[int]] = {}
            all_tids = req.all_token_ids
            bs = self._block_size
            for pos in range(total):
                start = pos * bs
                end = min(start + bs, len(all_tids))
                if start < len(all_tids):
                    block_token_ids[pos] = list(all_tids[start:end])

            requests_data.append({
                "req_id": req_id,
                "total_blocks": total,
                "num_computed_tokens": req.num_computed_tokens,
                "scores": scores,
                "num_protected_head": head_n,
                "num_protected_tail": tail_n,
                "evicted_positions": evicted,
                "block_token_ids": block_token_ids,
            })

        if requests_data:
            write_heatmap(requests_data, model_name=self._model_name)


def _safe_unlink(path: str) -> None:
    """Unlink ``path`` ignoring missing-file errors.

    Runs on the swap-unlink threadpool — must not raise, otherwise the
    next submission's exception logs the trace.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("kv-swap unlink failed: %s (%s)", path, e)
