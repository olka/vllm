# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.utils import config


@config
class KVCacheEvictionConfig:
    """Configuration for KV cache page eviction.

    When enabled, the scheduler evicts cold KV cache blocks under memory
    pressure instead of fully preempting requests. This avoids the
    "stop-the-world" cost of full preemption for long-context requests.

    Treats KV cache blocks as virtual memory pages: cold pages are evicted
    (paged out), and if needed again, recomputed via page fault handling.
    """

    enable: bool = False
    """Enable KV cache page eviction."""

    num_protected_head_blocks: int = 0
    """Number of head blocks to protect from eviction. Disabled by default
    (0) because the leading positions of any sequence become attention
    sinks (Xiao et al., ICLR 2024) and naturally score high under Q@K,
    so Otsu's selection retains them without explicit protection. Set
    to a small positive integer (e.g., 4) to re-enable for ablation or
    workloads where attention-sink behavior is not robust."""

    num_protected_tail_blocks: int = 2
    """Number of tail blocks to protect from eviction (recent context).
    These blocks contain the most recent tokens needed for coherent
    generation."""

    max_eviction_fraction: float = 0.25
    """Maximum fraction of a request's blocks that can be evicted per step.
    Prevents over-eviction that could cause quality collapse."""

    page_fault_usage_threshold: float = 0.90
    """Upper block pool usage bound for page fault recovery. Above this,
    evicted blocks are left as-is — recomputation needs spare blocks and we
    can't afford it under memory pressure."""

    page_fault_min_usage_threshold: float = 0.70
    """Lower block pool usage bound for page fault recovery. Below this,
    rollback is suppressed because there is no real memory pressure to justify
    undoing eviction. Proactive eviction (triggered by per-request context
    fill, not GPU pressure) lives below this threshold by design — without
    this gate, every proactive eviction is immediately reverted, defeating
    the eviction.

    Effective rollback band:
        page_fault_min_usage_threshold ≤ block_pool_usage ≤ page_fault_usage_threshold
    Set to 0.0 to roll back unconditionally below the upper bound (legacy
    behavior); set to 1.0 to disable rollback entirely (lossy mode)."""

    scoring_strategy: str = "paged_eviction"
    """Name of an EvictionStrategy subclass (registered via __init_subclass__).
    Built-in strategies:
      - 'paged_eviction': Q@K attention scoring with ||V||/||K|| fallback
      - 'triattention': trigonometric KV compression (stub, requires calibration)
    Custom strategies can be added by subclassing EvictionStrategy."""

    min_blocks_for_eviction: int = 8
    """Minimum number of blocks a request must have to be an eviction victim.
    Prevents eviction from thrashing on small requests."""

    proactive_eviction_threshold: float = 0.70
    """Per-request context window fill ratio at which proactive eviction
    starts. When a request's num_computed_tokens / max_model_len exceeds
    this threshold, the scheduler evicts cold blocks from that request
    to prevent full preemption. This is the "minor GC" that prevents the
    "major GC" (full preemption / stop-the-world compaction)."""

    enable_otsu_threshold: bool = False
    """Use Otsu's method to derive the dead-block threshold from the EMA
    score histogram each scoring cycle. When False, falls back to the
    strategy's static `_DEAD_THRESHOLD` ClassVar.

    Motivated by the empirical observation that bimodal separation
    intensifies with trace length: a fixed threshold over-evicts at long
    traces and under-evicts at short ones. Otsu adapts per cycle."""

    enable_swap: bool = False
    """Persist evicted KV blocks to disk (swap) instead of dropping them.
    When False (default), eviction frees the GPU block and recovery is
    via recompute (chunked prefill rollback). When True, the worker
    copies the block's KV bytes to ``swap_dir`` before the GPU block is
    returned to the pool, and recovery on miss reloads from disk."""

    enable_proactive_recovery: bool = False
    """Enable the proactive page-fault handler that eagerly recovers
    evicted blocks (swap-in / rollback) every scheduler cycle while
    pool usage is in the rollback band.

    Default False (paper-aligned). The current implementation has no
    demand signal — it cannot tell whether the model actually needs an
    evicted position, so it just picks the earliest entry from
    ``evicted_blocks`` and tries to recover it. With ``null_block``
    zeroing evicted positions, the model has no observable way to
    "demand" a block; the eager recovery thrashes (PAPER_DRAFT.md §6.1
    discussion of demand paging vs. the implementation's behavior).

    TODO/FIXME: replace with real demand-paged recovery — either
    shadow scoring against an evicted-block K summary (the C-shadow
    design discussed in conversation), or per-backend attention-kernel
    plumbing for an in-kernel "would have attended" signal. Until
    then, leave proactive recovery off; reactive recovery via
    ``try_free_blocks`` (the near-preemption path) remains active and
    handles the only legitimate intra-request swap-in case."""

    swap_dir: str = "/tmp/vllm_kv_swap"
    """Filesystem path for swapped-out KV blocks. One file per
    (request, kv_cache_group, block_position). The directory is created
    on first swap-out and is not durable across process restarts —
    contents are dropped on request finish, and orphaned files at startup
    are ignored."""

    cpu_tier_capacity_bytes: int = 0
    """Optional pinned-CPU swap tier (L2) sitting above the disk tier (L3).
    When > 0, evicted blocks land first in a bounded pinned-host pool of
    this many bytes; the oldest entries demote to ``swap_dir`` when the
    pool is over capacity. Recovery on miss hits the CPU tier first
    (~5–15 μs/block) before falling through to disk (~50–200 μs/block).
    Set to 0 to disable the CPU tier and write evictions directly to
    disk — useful on hardware where host DRAM is unified with GPU memory
    (GB10, MI300, Grace Hopper) and the CPU tier offers no real capacity
    advantage. Set to e.g. 8589934592 (8 GiB) on hardware with abundant
    host DRAM."""
