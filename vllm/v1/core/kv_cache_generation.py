# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generational eviction state for KV cache blocks.

Implements the "weak generational hypothesis" applied to KV cache:
recently-allocated blocks are most likely to die soon, while blocks
that survive several scoring sweeps tend to keep surviving. We track
three generations:

  - NURSERY: just allocated, or recently demoted. Default eviction
    target. Most blocks die here.
  - MATURE: survived ``nursery_to_mature_sweeps`` consecutive sweeps
    above the liveness threshold. Eviction candidate only when the
    nursery is exhausted.
  - TENURED: survived ``mature_to_tenured_sweeps`` further sweeps.
    Skipped during minor sweeps. Reclaimed only under heavy pressure.

Liveness is determined per-cycle by Otsu thresholding on the score
distribution (see :mod:`vllm.v1.core.otsu`). The scorer that produces
the per-block scores is orthogonal — any
:class:`~vllm.v1.core.kv_cache_eviction_strategy.EvictionStrategy`
plugs in here unchanged.

NOTE: This module owns *bookkeeping*. It does not perform eviction
itself; the manager consults the tracker to scope candidate sets and
applies its own head/tail protection and ``max_eviction_fraction`` cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Generation(IntEnum):
    """Block generation labels, ordered by survival.

    Integer-valued so callers can compare with ``<``/``>`` to express
    "evict everything below MATURE" without enum gymnastics.
    """

    NURSERY = 0
    MATURE = 1
    TENURED = 2


@dataclass
class GenerationalBlock:
    """Per-block generational metadata.

    Lives in :class:`GenerationTracker`'s nested dict, keyed by
    (request_id, block_position). Created on first scoring observation;
    dropped when the block is evicted or its request finishes.
    """

    generation: Generation = Generation.NURSERY
    #: Number of consecutive sweeps in which this block scored at or
    #: above the Otsu threshold. Reset to 0 on any below-threshold
    #: observation (which also demotes to NURSERY).
    survived_sweeps: int = 0
    #: Last EMA score absorbed for this block. None means we have not
    #: observed a worker score yet — the block is alive by default
    #: (treated as MATURE-equivalent for protection purposes during
    #: the worker-scoring warm-up window).
    last_score: float | None = None


@dataclass
class PromotionPolicy:
    """Knobs controlling the promotion ladder.

    Default values follow the classic generational-GC pattern: blocks
    are promoted aggressively out of the nursery, conservatively into
    the tenured population.
    """

    nursery_to_mature_sweeps: int = 1
    mature_to_tenured_sweeps: int = 4
    #: When True, a single below-threshold observation demotes a block
    #: all the way back to NURSERY. When False, demotion is one
    #: generation per below-threshold observation. The paper's claims
    #: assume aggressive demotion (True) since GC analogues never
    #: "partially" demote.
    aggressive_demotion: bool = True


class GenerationTracker:
    """Tracks per-block generational state across scoring cycles.

    Lifecycle:

      1. ``absorb_scores(...)`` is called by the manager once per
         scoring cycle. It computes the Otsu threshold from the score
         population and updates each block's ``survived_sweeps`` /
         ``generation`` accordingly.
      2. ``candidates_in(request_id, max_generation)`` returns block
         positions whose generation is at or below the given cap. The
         manager iterates ``NURSERY → MATURE`` until enough victims
         are found; ``TENURED`` is queried only by an explicit major
         sweep (not yet wired).
      3. ``forget(request_id)`` is called when a request finishes.
      4. ``forget_block(request_id, position)`` is called when a single
         block is evicted, to keep the tracker's view consistent.

    The tracker stores no scores itself — those live in the manager's
    ``_score_cache``. The tracker only sees them at absorption time.
    """

    def __init__(self, policy: PromotionPolicy | None = None) -> None:
        self._policy = policy or PromotionPolicy()
        # Maps req_id -> {block_position: GenerationalBlock}.
        self._state: dict[str, dict[int, GenerationalBlock]] = {}
        # Last computed Otsu threshold; exposed for logging / heatmap.
        self._last_threshold: float = 0.0

    # ─── Public API ──────────────────────────────────────────────────

    @property
    def last_threshold(self) -> float:
        return self._last_threshold

    def absorb_scores(
        self,
        scores_by_req: dict[str, dict[int, float]],
        threshold: float,
    ) -> None:
        """Update generation state given fresh per-block scores.

        ``threshold`` is computed by the caller (typically via
        :func:`~vllm.v1.core.otsu.otsu_threshold` over the union of all
        scores). Passing it in (rather than computing it here) lets the
        caller decide whether to use a global threshold across all
        requests or a per-request one — both are reasonable.
        """
        self._last_threshold = threshold
        policy = self._policy

        for req_id, per_block_scores in scores_by_req.items():
            req_state = self._state.setdefault(req_id, {})
            for pos, score in per_block_scores.items():
                meta = req_state.get(pos)
                if meta is None:
                    meta = GenerationalBlock()
                    req_state[pos] = meta
                meta.last_score = score

                if score >= threshold:
                    meta.survived_sweeps += 1
                    self._maybe_promote(meta)
                else:
                    if policy.aggressive_demotion:
                        meta.survived_sweeps = 0
                        meta.generation = Generation.NURSERY
                    else:
                        meta.survived_sweeps = 0
                        if meta.generation > Generation.NURSERY:
                            meta.generation = Generation(
                                meta.generation - 1
                            )

    def candidates_in(
        self,
        request_id: str,
        max_generation: Generation,
    ) -> list[int]:
        """Block positions for a request whose generation ≤ cap.

        Returns positions in arbitrary order. Caller applies its own
        head/tail protection, scoring, and eviction-fraction cap on
        top.
        """
        req_state = self._state.get(request_id)
        if not req_state:
            return []
        return [
            pos for pos, meta in req_state.items()
            if meta.generation <= max_generation
        ]

    def generation_of(
        self, request_id: str, position: int
    ) -> Generation:
        """Look up generation of a (req_id, pos). NURSERY by default."""
        req_state = self._state.get(request_id)
        if not req_state:
            return Generation.NURSERY
        meta = req_state.get(position)
        return meta.generation if meta is not None else Generation.NURSERY

    def forget(self, request_id: str) -> None:
        self._state.pop(request_id, None)

    def forget_block(self, request_id: str, position: int) -> None:
        req_state = self._state.get(request_id)
        if req_state is not None:
            req_state.pop(position, None)
            if not req_state:
                self._state.pop(request_id, None)

    def histogram(self) -> dict[Generation, int]:
        """Population by generation across all tracked blocks.

        Useful for paper figures / KVCacheVis annotations.
        """
        counts = {g: 0 for g in Generation}
        for req_state in self._state.values():
            for meta in req_state.values():
                counts[meta.generation] += 1
        return counts

    # ─── Internal ────────────────────────────────────────────────────

    def _maybe_promote(self, meta: GenerationalBlock) -> None:
        policy = self._policy
        if (meta.generation == Generation.NURSERY
                and meta.survived_sweeps >= policy.nursery_to_mature_sweeps):
            meta.generation = Generation.MATURE
            return
        if (meta.generation == Generation.MATURE
                and meta.survived_sweeps >=
                (policy.nursery_to_mature_sweeps
                 + policy.mature_to_tenured_sweeps)):
            meta.generation = Generation.TENURED
