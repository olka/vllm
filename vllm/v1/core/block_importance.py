# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block importance scoring for KV cache page eviction.

Implements two complementary scoring methods:
- PagedEviction: Static ||V||/||K|| norm-ratio proxy (zero overhead)
- Quest: Channel-wise min/max Key stats for query-aware page fault prediction
"""

from dataclasses import dataclass

import torch


@dataclass
class QuestBlockStats:
    """Per-block min/max Key statistics for Quest-style page fault prediction.

    Stored in CPU memory after a block is evicted. Small footprint:
    ~2 * num_kv_heads * head_size floats per evicted block.
    """

    # Channel-wise min/max of Key vectors in this block.
    # Shape: [num_kv_heads, head_size]
    key_min: torch.Tensor
    key_max: torch.Tensor

    # Token IDs that were in this block (needed for recomputation).
    token_ids: list[int]

    # Position range [start, end) in the original sequence.
    position_start: int
    position_end: int

    # The block's original position index in req_to_blocks.
    block_position: int


@dataclass
class EvictedBlockInfo:
    """Metadata for a single evicted block, enabling page fault prediction
    and recomputation."""

    quest_stats: QuestBlockStats
    # Which KV cache group this block belonged to.
    kv_cache_group_id: int


class BlockImportanceScorer:
    """Computes per-block importance scores using the PagedEviction norm-ratio.

    PagedEviction (arXiv 2509.04377):
        block_score = mean(||V_i||_2 / ||K_i||_2)

    Key insight: Key L2-norm is inversely proportional to cumulative attention
    weight. Tokens with large Key norms receive less attention, so blocks with
    high V/K ratio contain tokens that contribute more value per unit of
    attention received.

    Lower score = less important = evict first.
    """

    @staticmethod
    def score_blocks(
        kv_cache: torch.Tensor,
        block_ids: list[int],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        """Compute PagedEviction importance scores for the given blocks.

        Args:
            kv_cache: The KV cache tensor for one layer.
                Shape: (2, num_blocks, block_size, num_kv_heads, head_size)
                where dim 0: 0=key, 1=value.
            block_ids: List of block IDs to score.
            block_size: Number of tokens per block.
            num_kv_heads: Number of KV attention heads.
            head_size: Dimension of each attention head.

        Returns:
            Tensor of shape [len(block_ids)] with importance scores.
            Lower score = less important = evict first.
        """
        if not block_ids:
            return torch.tensor([], device=kv_cache.device)

        block_id_tensor = torch.tensor(
            block_ids, device=kv_cache.device, dtype=torch.long
        )

        # Extract K and V for the requested blocks.
        # kv_cache shape: (2, num_blocks, block_size, num_kv_heads, head_size)
        key_blocks = kv_cache[0, block_id_tensor]  # (N, block_size, H, D)
        val_blocks = kv_cache[1, block_id_tensor]  # (N, block_size, H, D)

        # Compute per-token L2 norms across head_size dimension.
        # Shape: (N, block_size, num_kv_heads)
        key_norms = torch.linalg.norm(
            key_blocks.float(), dim=-1
        )
        val_norms = torch.linalg.norm(
            val_blocks.float(), dim=-1
        )

        # Avoid division by zero for empty/uninitialized slots.
        key_norms = key_norms.clamp(min=1e-8)

        # Per-token V/K ratio, then mean across tokens and heads per block.
        # Shape: (N, block_size, num_kv_heads) -> (N,)
        ratios = val_norms / key_norms
        scores = ratios.mean(dim=(1, 2))

        return scores

    @staticmethod
    def score_blocks_by_attention(
        kv_cache: torch.Tensor,
        query: torch.Tensor,
        block_ids: list[int],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        """Score blocks by actual attention: softmax(Q @ K^T / sqrt(d)).

        Computes per-block attention mass from a single query token against
        all key tokens in the given blocks. Higher score = more attended =
        more important.

        Args:
            kv_cache: KV cache tensor for one layer.
                Shape: (2, num_blocks, block_size, num_kv_heads, head_size)
            query: Query vector for one token.
                Shape: (num_q_heads, head_size)
            block_ids: Block IDs to score.
            block_size: Tokens per block.
            num_kv_heads: Number of KV attention heads.
            head_size: Head dimension.

        Returns:
            Tensor of shape [len(block_ids)] with per-block attention mass.
            Higher score = more important.
        """
        if not block_ids:
            return torch.tensor([], device=kv_cache.device)

        num_q_heads = query.shape[0]
        group_size = num_q_heads // num_kv_heads

        block_id_tensor = torch.tensor(
            block_ids, device=kv_cache.device, dtype=torch.long
        )

        # K blocks: [N, block_size, num_kv_heads, head_size]
        # Stay in cache dtype (BF16/FP16) — scoring doesn't need FP32
        # precision, and this halves the memory bandwidth for K gather.
        key_blocks = kv_cache[0, block_id_tensor]
        q = query.to(key_blocks.dtype)

        # Handle GQA: average Q heads within each KV group.
        if group_size > 1:
            q = q.view(num_kv_heads, group_size, head_size).mean(dim=1)
        # q: [num_kv_heads, head_size]

        N = len(block_ids)
        # Reshape K: [num_kv_heads, N * block_size, head_size]
        k = key_blocks.permute(2, 0, 1, 3).reshape(
            num_kv_heads, N * block_size, head_size
        )

        # Q @ K^T / sqrt(d): [num_kv_heads, 1, N * block_size]
        scale = head_size**-0.5
        attn = torch.bmm(
            q.unsqueeze(1), k.transpose(1, 2)
        ) * scale

        # Softmax across all positions.
        attn = torch.softmax(attn, dim=-1)

        # Sum per block across heads and tokens: [N]
        attn = attn.view(num_kv_heads, N, block_size)
        scores = attn.sum(dim=(0, 2))

        return scores

    @staticmethod
    def capture_quest_stats(
        kv_cache: torch.Tensor,
        block_id: int,
        token_ids: list[int],
        position_start: int,
        position_end: int,
        block_position: int,
    ) -> QuestBlockStats:
        """Capture Quest min/max Key statistics before evicting a block.

        These stats are stored in CPU memory and used by PageFaultPredictor
        to determine if an evicted block needs recomputation.

        Args:
            kv_cache: The KV cache tensor for one layer.
                Shape: (2, num_blocks, block_size, num_kv_heads, head_size)
            block_id: The block ID to capture stats for.
            token_ids: Token IDs stored in this block.
            position_start: Start position in the sequence.
            position_end: End position in the sequence (exclusive).
            block_position: Index in the request's block list.

        Returns:
            QuestBlockStats with min/max Key vectors on CPU.
        """
        # Extract Key vectors for this block.
        # Shape: (block_size, num_kv_heads, head_size)
        key_block = kv_cache[0, block_id].float()

        # Only use positions that have actual tokens.
        num_tokens = position_end - position_start
        key_block = key_block[:num_tokens]

        # Channel-wise min/max across the token dimension.
        # Shape: (num_kv_heads, head_size)
        key_min = key_block.min(dim=0).values.cpu()
        key_max = key_block.max(dim=0).values.cpu()

        return QuestBlockStats(
            key_min=key_min,
            key_max=key_max,
            token_ids=token_ids,
            position_start=position_start,
            position_end=position_end,
            block_position=block_position,
        )

    @staticmethod
    def select_blocks_to_evict(
        scores: torch.Tensor,
        block_indices: list[int],
        num_to_evict: int,
        num_protected_head: int = 4,
        num_protected_tail: int = 2,
        total_blocks: int = 0,
    ) -> list[int]:
        """Select which blocks to evict based on importance scores.

        Protects the first N blocks (attention sinks / system prompt) and
        the last M blocks (recent context).

        Args:
            scores: Importance scores, one per block in block_indices.
            block_indices: The position indices of blocks in the request's
                block list (not block IDs).
            num_to_evict: How many blocks to evict.
            num_protected_head: Number of head blocks to protect.
            num_protected_tail: Number of tail blocks to protect.
            total_blocks: Total number of blocks for the request
                (needed to determine tail protection range).

        Returns:
            List of block position indices to evict (subset of block_indices),
            sorted ascending.
        """
        if total_blocks == 0:
            total_blocks = max(block_indices) + 1 if block_indices else 0

        # Filter out protected blocks.
        tail_start = total_blocks - num_protected_tail
        candidates = []
        candidate_scores = []
        for idx, score in zip(block_indices, scores.tolist()):
            if idx < num_protected_head:
                continue
            if idx >= tail_start:
                continue
            candidates.append(idx)
            candidate_scores.append(score)

        if not candidates:
            return []

        # Sort by score ascending (lowest = least important = evict first).
        num_to_evict = min(num_to_evict, len(candidates))
        scored = sorted(zip(candidate_scores, candidates))
        evict_indices = sorted([idx for _, idx in scored[:num_to_evict]])
        return evict_indices
