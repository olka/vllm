# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable scoring strategies for KV cache page eviction.

Each strategy decides three things:

  - Whether the worker needs to compute importance scores from the
    GPU-resident KV cache (`needs_worker_scoring`). If True, the worker
    calls `compute_worker_scores` periodically; the manager surfaces the
    results via `worker_score` in `score_block`.
  - How to rank an evictable block (`score_block`). Lower = more
    evictable. The manager applies head/tail protection and the
    `max_eviction_fraction` cap on top.
  - Which blocks are "dead enough" for proactive eviction (`is_dead`).
    Strategies that only do reactive eviction return False.

Subclasses auto-register by name via `__init_subclass__`. To add a custom
strategy, define a subclass with a non-empty `name` ClassVar — that
single act registers it. No plugin manifest, no entry points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch

    from vllm.config.kv_cache_eviction import KVCacheEvictionConfig
    from vllm.v1.core.block_pool import KVCacheBlock


class EvictionStrategy(ABC):
    """Base class for KV cache page eviction scoring strategies.

    All instances are constructed via `EvictionStrategy.create(name, ...)`,
    which routes to the registered subclass for `name`. Custom strategies
    register themselves simply by subclassing — no manifest required.
    """

    #: Registered name. Empty for the abstract base; subclasses MUST set it.
    name: ClassVar[str] = ""

    #: If True, the manager periodically asks the worker to compute scores
    #: by calling :meth:`compute_worker_scores` on the GPU.
    needs_worker_scoring: ClassVar[bool] = False

    #: Auto-populated by `__init_subclass__`.
    _registry: ClassVar[dict[str, type["EvictionStrategy"]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            EvictionStrategy._registry[cls.name] = cls

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def create(cls, name: str, **kwargs: object) -> "EvictionStrategy":
        """Instantiate a registered strategy by name."""
        impl = cls._registry.get(name)
        if impl is None:
            raise ValueError(
                f"Unknown eviction strategy {name!r}. "
                f"Registered: {sorted(cls._registry)}"
            )
        return impl(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def registered(cls) -> list[str]:
        return sorted(cls._registry)

    # ── Manager-side API ─────────────────────────────────────────────

    @abstractmethod
    def score_block(
        self,
        position: int,
        block: "KVCacheBlock",
        worker_score: float | None,
    ) -> float:
        """Per-block importance. Lower score = more evictable.

        `worker_score` is the EMA-smoothed value the worker last reported
        for this block, or None if no observation is available yet.
        """

    def is_dead(self, score: float, dead_threshold: float) -> bool:
        """Whether the block is dead enough for proactive eviction.

        The manager owns the threshold (static fallback or Otsu-derived,
        depending on `enable_otsu_threshold`). Strategies override only
        if they want non-threshold dead semantics (e.g., score-and-age).

        Default: never (returns False regardless of threshold), so
        strategies that don't opt into proactive eviction don't have to
        override this.
        """
        return False

    # ── Worker-side API ──────────────────────────────────────────────

    def compute_worker_scores(
        self,
        kv_cache: "torch.Tensor",
        block_ids: list[int],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        query: "torch.Tensor | None",
    ) -> "torch.Tensor":
        """Compute scores for `block_ids` from the worker's KV cache.

        Called from the GPU model runner when this strategy declares
        `needs_worker_scoring = True`. Strategies that don't need worker
        compute leave this unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares needs_worker_scoring="
            f"{self.needs_worker_scoring} but does not implement "
            "compute_worker_scores"
        )


class PagedEvictionStrategy(EvictionStrategy):
    """PagedEviction (arXiv 2509.04377): rank by ||V|| / ||K|| norm-ratio.

    Worker computes scores per block from the live KV cache. Falls back
    to LRU when no worker score is available yet (warm-up period).
    Proactive eviction triggers when the EMA score drops below
    :attr:`_DEAD_THRESHOLD`.

    The worker-side compute also handles the Q@K attention variant — when
    a captured query is provided, the score reflects actual attention
    mass rather than the static V/K proxy.
    """

    name: ClassVar[str] = "paged_eviction"
    needs_worker_scoring: ClassVar[bool] = True

    #: Empirical absolute threshold below which an EMA Q@K / V/K score is
    #: dead. KV attention is bimodal — well-separated absolute thresholds
    #: work better than relative ones. Will be superseded by Otsu in
    #: generational mode (see GENERATIONAL_DESIGN.pyi).
    _DEAD_THRESHOLD: ClassVar[float] = 1e-4

    def score_block(
        self,
        position: int,
        block: "KVCacheBlock",
        worker_score: float | None,
    ) -> float:
        if worker_score is None:
            # No observation yet — fall back to LRU. Means we order
            # by allocation time during the first ~50 steps.
            return float(block.last_accessed)
        return worker_score

    def is_dead(self, score: float, dead_threshold: float) -> bool:
        return score < dead_threshold

    def compute_worker_scores(
        self,
        kv_cache: "torch.Tensor",
        block_ids: list[int],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        query: "torch.Tensor | None",
    ) -> "torch.Tensor":
        from vllm.v1.core.block_importance import BlockImportanceScorer

        if query is not None:
            return BlockImportanceScorer.score_blocks_by_attention(
                kv_cache=kv_cache,
                query=query,
                block_ids=block_ids,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
            )
        return BlockImportanceScorer.score_blocks(
            kv_cache=kv_cache,
            block_ids=block_ids,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )


class TriAttentionStrategy(EvictionStrategy):
    """TriAttention (arXiv 2604.04921): trigonometric KV compression.

    Uses pre-RoPE Q/K concentration to score keys via a trigonometric
    series of distance preferences, plus a norm fallback weighted by
    (1 - R_f) per frequency band. Calibration produces per-head Q-centers
    and concentration ratios; scoring is value-only (no live query).

    NOTE: This is a stub. Calibration loading and the trig-series
    scoring kernel are not yet implemented. Activating this strategy
    will raise NotImplementedError at the first worker-scoring call.
    """

    name: ClassVar[str] = "triattention"
    needs_worker_scoring: ClassVar[bool] = True

    #: Placeholder dead-block threshold. TriAttention scores live in a
    #: different regime than PagedEviction (z-normed across GQA groups,
    #: paper Eq. 12-13) so this likely needs a TriAttention-specific
    #: value once calibration is wired. Until then, reuse the
    #: PagedEviction empirical value.
    _DEAD_THRESHOLD: ClassVar[float] = 1e-4

    def __init__(self, calibration_path: str | None = None) -> None:
        # Path to the precomputed Q/K frequency stats `.pt` file.
        # See https://github.com/WeianMao/triattention/blob/main/docs/calibration.md
        # for the file format produced by their `scripts/calibrate.py`.
        self._calibration_path = calibration_path
        # Loaded calibration tensors:
        #   per-head Q-centers (E[q_f]), Q-norm means (E[||q_f||]),
        #   concentration ratios R_f. Shape depends on model.
        self._stats: dict[str, "torch.Tensor"] | None = None
        if calibration_path is not None:
            self._stats = self._load_calibration(calibration_path)

    def _load_calibration(
        self, path: str
    ) -> dict[str, "torch.Tensor"]:
        raise NotImplementedError(
            "TriAttention calibration loading is not yet implemented. "
            "Expected format: torch.load() returns a dict with per-layer "
            "tensors {q_center, q_norm_mean, R_per_band}."
        )

    def score_block(
        self,
        position: int,
        block: "KVCacheBlock",
        worker_score: float | None,
    ) -> float:
        # Fall back to LRU until the worker has produced a real score.
        if worker_score is None:
            return float(block.last_accessed)
        return worker_score

    def is_dead(self, score: float, dead_threshold: float) -> bool:
        return score < dead_threshold

    def compute_worker_scores(
        self,
        kv_cache: "torch.Tensor",
        block_ids: list[int],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        query: "torch.Tensor | None",
    ) -> "torch.Tensor":
        # Stub. The real implementation needs:
        #   1. Per-block ||K_f|| computed across RoPE frequency bands f
        #      (post-RoPE K from the cache, decomposed into pairs).
        #   2. S_trig(k, Δ) = Σ_f ||E[q_f]||·||k_f||·cos(ω_f Δ + φ_f)
        #      averaged over the geometric offset set D = {1, 2, 4, …, 2^16}
        #      (paper Eq. 6, 11).
        #   3. S_norm(k) = Σ_f (1 - R_f)·E[||q_f||]·||k_f||  (paper Eq. 9)
        #   4. Z-norm within GQA query heads, max across heads
        #      (paper Eq. 12-13).
        # Calibration tensors (self._stats) supply E[q_f], E[||q_f||], R_f.
        raise NotImplementedError(
            "TriAttentionStrategy.compute_worker_scores is a stub. "
            "Implement the trigonometric series scoring (paper Eq. 6-13)."
        )
