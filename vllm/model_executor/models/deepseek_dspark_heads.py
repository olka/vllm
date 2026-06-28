# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark sequential heads: low-rank Markov transition bias + confidence head.

These are the two pieces DSpark adds on top of the (DFlash-style) parallel block
backbone — see arXiv 2606.19348, Eq. 5 (Markov head) and Eq. 7 (confidence head).

Both are deliberately framework-light (plain ``torch.nn``, no vLLM layer deps) so the
numerics can be unit-tested in isolation: the parallel backbone runs once and produces
base logits ``U_1..U_gamma``; the Markov head then injects intra-block dependency purely
in logit space, and the confidence head scores each position for prefix truncation.

Checkpoint shapes (DeepSeek-V4-Flash-DSpark, ``mtp.2.*``):
  markov_head.markov_w1   [vocab, rank]   e.g. [129280, 256]
  markov_head.markov_w2   [vocab, rank]   e.g. [129280, 256]
  confidence_head.proj    [1, hidden+rank] e.g. [1, 4352]   (4096 + 256, no bias)
"""

from __future__ import annotations

import torch
from torch import nn


class MarkovHead(nn.Module):
    """Low-rank first-order transition bias ``B(x) = W1[x] @ W2.T`` (paper Eq. 5).

    The full ``vocab x vocab`` transition matrix is factorized at ``rank`` (256 by
    default): ``markov_w1`` is an embedding lookup of the previously sampled token into
    rank space, ``markov_w2`` projects rank space back to vocab logits. The bias is added
    to the backbone's base logit before sampling each block position, which is what
    breaks the parallel drafter's mode-collision / suffix decay.
    """

    def __init__(self, vocab_size: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = rank
        # Stored as [vocab, rank] (embedding-style) — matching the checkpoint, so both
        # factors share the same lookup-friendly layout.
        self.markov_w1 = nn.Parameter(torch.empty(vocab_size, rank))
        self.markov_w2 = nn.Parameter(torch.empty(vocab_size, rank))

    def prev_token_embedding(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """``W1[x]`` — [..., rank]. Also reused by the confidence head."""
        return self.markov_w1[prev_token_ids]

    def bias_from_embedding(self, w1_x: torch.Tensor) -> torch.Tensor:
        """Project a rank-space vector to a full vocab bias: ``w1_x @ W2.T`` -> [..., V]."""
        return torch.matmul(w1_x, self.markov_w2.t())

    def bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """Transition bias ``B(x)`` for the given previous tokens -> [..., vocab]."""
        return self.bias_from_embedding(self.prev_token_embedding(prev_token_ids))


class ConfidenceHead(nn.Module):
    """Per-position acceptance estimate ``c_k = sigmoid(w . [h_k ; W1[x_{k-1}]])``.

    Predicts the *conditional* probability that draft token ``k`` survives target
    verification given all earlier block tokens were accepted (paper Eq. 7). The input is
    the backbone hidden state concatenated with the Markov embedding of the previous
    token; the projection is bias-free and outputs a single logit per position.
    """

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size + rank, 1, bias=False)

    def logits(self, hidden: torch.Tensor, w1_prev: torch.Tensor) -> torch.Tensor:
        """Confidence logits -> [...]; sigmoid gives per-position survival probability."""
        return self.proj(torch.cat([hidden, w1_prev], dim=-1)).squeeze(-1)


def markov_block_sample(
    base_logits: torch.Tensor,
    anchor_token_id: torch.Tensor,
    markov: MarkovHead,
    *,
    greedy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-to-right block sampling with the Markov bias (paper Eq. 4-5).

    The backbone is run once to produce ``base_logits`` ``U_1..U_gamma`` for the whole
    block; here we walk the block sequentially, adding ``B(x_{k-1})`` before sampling
    position ``k``. This loop touches only logits (no transformer re-run), so it stays in
    the ``T_sequential << T_parallel`` regime. It is a fixed-length, CUDA-graph-friendly
    unroll.

    Args:
        base_logits:   [gamma, vocab] backbone base logits for one request.
        anchor_token_id: scalar — the bonus/anchor token ``x_0`` from the previous round.
        markov:        the Markov head.
        greedy:        argmax (used for verification-exact draft probs) vs multinomial.

    Returns:
        draft_token_ids:  [gamma]
        adjusted_logits:  [gamma, vocab] — ``U_k + B(x_{k-1})``. These (not the base
                          logits) are the draft distribution handed to the rejection
                          sampler, so verification stays exact/lossless.
    """
    gamma = base_logits.shape[0]
    prev = anchor_token_id.reshape(())
    tokens: list[torch.Tensor] = []
    adjusted: list[torch.Tensor] = []
    for k in range(gamma):
        logit_k = base_logits[k] + markov.bias(prev)
        adjusted.append(logit_k)
        if greedy:
            x_k = logit_k.argmax(dim=-1)
        else:
            x_k = torch.multinomial(logit_k.softmax(dim=-1), num_samples=1).squeeze(-1)
        tokens.append(x_k)
        prev = x_k
    return torch.stack(tokens), torch.stack(adjusted)
