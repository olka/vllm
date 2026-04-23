# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

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

    num_protected_head_blocks: int = 4
    """Number of head blocks to protect from eviction (attention sinks).
    These blocks typically contain the system prompt and initial context
    which carry outsized importance for attention."""

    num_protected_tail_blocks: int = 2
    """Number of tail blocks to protect from eviction (recent context).
    These blocks contain the most recent tokens needed for coherent
    generation."""

    max_eviction_fraction: float = 0.25
    """Maximum fraction of a request's blocks that can be evicted per step.
    Prevents over-eviction that could cause quality collapse."""

    quest_threshold: float = 0.1
    """Threshold for Quest-based page fault prediction. Evicted blocks with
    a Quest score above this threshold are predicted to be needed and will
    be proactively recomputed before the next forward pass."""

    page_fault_usage_threshold: float = 0.90
    """Maximum block pool usage ratio at which page fault recovery (rollback +
    recomputation) is attempted. Above this threshold, evicted blocks are left
    as-is to avoid memory pressure from recomputation."""

    scoring_strategy: Literal["access_time", "paged_eviction"] = "access_time"
    """Strategy for scoring block importance:
    - 'access_time': Evicts oldest-allocated blocks first (LRU-like)
    - 'paged_eviction': Uses ||V||/||K|| norm ratio from GPU cache data
    """

    min_blocks_for_eviction: int = 8
    """Minimum number of blocks a request must have to be an eviction victim.
    Prevents eviction from thrashing on small requests."""

    proactive_eviction_threshold: float = 0.70
    """Per-request context window fill ratio at which proactive eviction
    starts. When a request's num_computed_tokens / max_model_len exceeds
    this threshold, the scheduler evicts cold blocks from that request
    to prevent full preemption. This is the "minor GC" that prevents the
    "major GC" (full preemption / stop-the-world compaction)."""

    dead_block_epsilon: float = 1e-4
    """Absolute attention threshold below which a block is considered dead.
    Blocks whose EMA Q@K score stays below this value are evicted regardless
    of what other blocks score. Based on the observation that KV cache
    attention is bimodal (hot vs cold), not normally distributed — relative
    thresholds like 0.5*avg misrepresent both populations."""
