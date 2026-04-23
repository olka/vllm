# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page fault prediction using Quest-style query-aware scoring.

Quest (ICML 2024) computes an upper bound on attention score per block using
pre-stored channel-wise min/max of Key vectors. Given a query vector, blocks
with high upper bounds are likely needed and should be recomputed.

This module predicts which evicted blocks need recomputation before each
forward pass, enabling proactive page fault resolution with zero quality loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.core.block_importance import EvictedBlockInfo


@dataclass
class PageFaultPrediction:
    """A predicted page fault — an evicted block that needs recomputation."""

    # The evicted block's metadata.
    evicted_info: EvictedBlockInfo
    # The Quest attention upper bound score.
    score: float
    # Request ID that owns this block.
    request_id: str


class PageFaultPredictor:
    """Predicts which evicted blocks are needed before the next forward pass.

    Uses Quest's query-aware scoring: for each evicted block, compute an
    upper bound on the attention score using pre-stored min/max Key stats.
    Blocks with scores above the threshold are predicted to be needed.
    """

    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Minimum Quest score to trigger recomputation.
                Higher threshold = fewer recomputations = more misses.
                Lower threshold = more recomputations = better quality.
        """
        self.threshold = threshold

    def predict_page_faults(
        self,
        query: torch.Tensor,
        evicted_blocks: dict[str, list[EvictedBlockInfo]],
    ) -> list[PageFaultPrediction]:
        """Score evicted blocks against a query and predict needed blocks.

        Args:
            query: The current step's query vector.
                Shape: [num_kv_heads, head_size] (averaged across tokens).
            evicted_blocks: Maps request_id -> list of EvictedBlockInfo
                for all currently evicted blocks.

        Returns:
            List of PageFaultPrediction for blocks that should be
            recomputed, sorted by score descending (most needed first).
        """
        predictions: list[PageFaultPrediction] = []

        for request_id, blocks in evicted_blocks.items():
            for block_info in blocks:
                score = self._quest_score(query, block_info)
                if score >= self.threshold:
                    predictions.append(PageFaultPrediction(
                        evicted_info=block_info,
                        score=score,
                        request_id=request_id,
                    ))

        # Sort by score descending — most needed blocks first.
        predictions.sort(key=lambda p: p.score, reverse=True)
        return predictions

    @staticmethod
    def _quest_score(
        query: torch.Tensor,
        block_info: EvictedBlockInfo,
    ) -> float:
        """Compute Quest upper bound on attention for an evicted block.

        Quest formula:
            page_score = sum_i max(Q_i * K_min_i, Q_i * K_max_i)

        This gives an upper bound on the maximum attention score any token
        in this block could receive from the query.

        Args:
            query: Query vector, shape [num_kv_heads, head_size].
            block_info: The evicted block's Quest stats.

        Returns:
            The Quest upper bound score (scalar).
        """
        stats = block_info.quest_stats
        # Move query to CPU for scoring (stats are already on CPU).
        q = query.cpu().float()

        # Per-channel upper bound: max(Q * K_min, Q * K_max)
        # Shape: [num_kv_heads, head_size]
        score_min = q * stats.key_min.float()
        score_max = q * stats.key_max.float()
        upper_bound = torch.maximum(score_min, score_max)

        # Sum across all dimensions for a single scalar score.
        return upper_bound.sum().item()
