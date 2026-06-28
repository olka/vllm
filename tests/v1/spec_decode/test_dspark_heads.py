# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DSpark sequential heads (Markov bias + confidence).

These exercise pure-tensor numerics with no GPU/vLLM-runtime dependency, so they run on
CPU. The Markov head, confidence head, and the block-sampling loop are validated against
the checkpoint shapes of DeepSeek-V4-Flash-DSpark (vocab 129280, hidden 4096, rank 256).
"""

import torch

from vllm.model_executor.models.deepseek_dspark_heads import (
    ConfidenceHead,
    MarkovHead,
    markov_block_sample,
)
from vllm.v1.spec_decode.dspark import fit_linear_cost, schedule_prefixes

VOCAB, HIDDEN, RANK, GAMMA = 64, 32, 8, 5  # small stand-ins for fast CPU tests


def _markov() -> MarkovHead:
    torch.manual_seed(0)
    m = MarkovHead(VOCAB, HIDDEN, RANK)
    m.markov_w1.data.normal_()
    m.markov_w2.data.normal_()
    return m


class TestMarkovHead:
    def test_shapes(self):
        m = _markov()
        assert m.markov_w1.shape == (VOCAB, RANK)
        assert m.markov_w2.shape == (VOCAB, RANK)
        assert m.bias(torch.tensor(3)).shape == (VOCAB,)

    def test_bias_is_low_rank_factorization(self):
        """B(x) must equal W1[x] @ W2.T exactly (the rank-256 factorization)."""
        m = _markov()
        x = torch.tensor(7)
        ref = m.markov_w1[x] @ m.markov_w2.t()
        torch.testing.assert_close(m.bias(x), ref)

    def test_bias_batched(self):
        m = _markov()
        xs = torch.tensor([1, 2, 3])
        assert m.bias(xs).shape == (3, VOCAB)


class TestConfidenceHead:
    def test_input_width_is_hidden_plus_rank(self):
        c = ConfidenceHead(HIDDEN, RANK)
        assert c.proj.in_features == HIDDEN + RANK
        assert c.proj.out_features == 1
        assert c.proj.bias is None  # checkpoint has no confidence bias

    def test_logits_shape_and_sigmoid_range(self):
        c = ConfidenceHead(HIDDEN, RANK)
        h = torch.randn(GAMMA, HIDDEN)
        w1_prev = torch.randn(GAMMA, RANK)
        cl = c.logits(h, w1_prev)
        assert cl.shape == (GAMMA,)
        assert (cl.sigmoid() > 0).all() and (cl.sigmoid() < 1).all()


class TestMarkovBlockSample:
    def test_sequential_dependency_and_adjusted_logits(self):
        """Adjusted logit at position k must be U_k + B(x_{k-1}); greedy picks
        its argmax."""
        m = _markov()
        base = torch.randn(GAMMA, VOCAB)
        anchor = torch.tensor(11)
        tokens, adjusted = markov_block_sample(base, anchor, m, greedy=True)

        assert tokens.shape == (GAMMA,)
        assert adjusted.shape == (GAMMA, VOCAB)
        # Re-derive the recurrence independently.
        prev = anchor
        for k in range(GAMMA):
            expected_logit = base[k] + m.bias(prev)
            torch.testing.assert_close(adjusted[k], expected_logit)
            expected_tok = expected_logit.argmax()
            assert tokens[k] == expected_tok
            prev = tokens[k]

    def test_adjusted_logits_are_the_draft_distribution(self):
        """The returned logits (used for lossless verification) differ from the base
        logits whenever the Markov bias is non-trivial — i.e. dependency is really
        injected, not a no-op."""
        m = _markov()
        base = torch.randn(GAMMA, VOCAB)
        _, adjusted = markov_block_sample(base, torch.tensor(5), m, greedy=True)
        assert not torch.allclose(adjusted, base)

    def test_greedy_is_deterministic(self):
        m = _markov()
        base = torch.randn(GAMMA, VOCAB)
        t1, _ = markov_block_sample(base, torch.tensor(2), m, greedy=True)
        t2, _ = markov_block_sample(base, torch.tensor(2), m, greedy=True)
        torch.testing.assert_close(t1, t2)


# ---------------------------------------------------------------------------
# Hardware-aware prefix scheduler: cost-model fit + greedy budget allocation.
# ---------------------------------------------------------------------------
def test_fit_linear_cost_recovers_slope():
    # synthetic step-time T(B) = 0.1 + 0.002 * B
    samples = [(float(b), 0.1 + 0.002 * b) for b in range(10, 110, 10)]
    a, b = fit_linear_cost(samples)
    assert abs(a - 0.1) < 1e-6
    assert abs(b - 0.002) < 1e-6


def test_fit_linear_cost_degenerate():
    assert fit_linear_cost([]) is None
    assert fit_linear_cost([(10.0, 1.0)]) is None  # < 2 samples
    assert fit_linear_cost([(10.0, 1.0), (10.0, 2.0)]) is None  # no B variation
    # negative slope (faster with more tokens) is implausible -> rejected
    assert fit_linear_cost([(10.0, 2.0), (20.0, 1.0)]) is None


def test_schedule_prefixes_uncalibrated_keeps_full_block():
    surv = torch.rand(4, GAMMA).cumprod(dim=1)
    keep = schedule_prefixes(surv, cost_a=None, cost_b=None)
    assert torch.equal(keep, torch.full((4,), GAMMA, dtype=torch.int32))


def test_schedule_prefixes_memory_bound_keeps_full_block():
    surv = (0.5 * torch.ones(4, GAMMA)).cumprod(dim=1)
    # negligible per-token cost -> extra verification is ~free -> keep everything
    keep = schedule_prefixes(surv, cost_a=1.0, cost_b=1e-9)
    assert torch.equal(keep, torch.full((4,), GAMMA, dtype=torch.int32))


def test_schedule_prefixes_compute_bound_prunes():
    surv = (0.5 * torch.ones(4, GAMMA)).cumprod(dim=1)
    # large per-token cost -> each extra verify token is expensive -> prune
    keep = schedule_prefixes(surv, cost_a=1.0, cost_b=10.0)
    assert int(keep.max()) < GAMMA
    assert int(keep.min()) >= 1  # every request drafts at least one token


def test_schedule_prefixes_confidence_ordered_and_valid():
    surv = torch.tensor(
        [
            [0.95, 0.90, 0.85, 0.80, 0.75],  # high-confidence request
            [0.40, 0.16, 0.06, 0.02, 0.01],  # low-confidence request
        ]
    )
    keep = schedule_prefixes(surv, cost_a=1.0, cost_b=0.5)
    assert keep.shape == (2,)
    assert (keep >= 1).all() and (keep <= GAMMA).all()
    # budget flows to the higher-survival request first
    assert int(keep[0]) >= int(keep[1])


def test_schedule_prefixes_batch_one():
    surv = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.1]])  # R=1
    keep = schedule_prefixes(surv, cost_a=1.0, cost_b=0.5)
    assert keep.shape == (1,)
    assert 1 <= int(keep[0]) <= GAMMA


def test_schedule_prefixes_all_one_survival_keeps_full():
    # every position certain to survive + ~free verification -> keep the block
    surv = torch.ones(3, GAMMA)
    keep = schedule_prefixes(surv, cost_a=1.0, cost_b=1e-6)
    assert torch.equal(keep, torch.full((3,), GAMMA, dtype=torch.int32))


def test_schedule_prefixes_all_zero_survival_keeps_min_one():
    # nothing survives, but every request must still draft >=1 token: an empty
    # (zero-length) draft yields ragged shapes that crash the verify path at high
    # concurrency, so the scheduler floors keep at 1.
    surv = torch.zeros(3, GAMMA)
    keep = schedule_prefixes(surv, cost_a=1.0, cost_b=5.0)
    assert torch.equal(keep, torch.ones(3, dtype=torch.int32))


def test_schedule_prefixes_kstar_zero_under_huge_cost():
    # huge per-token cost -> optimum verifies no draft tokens (kstar=0), but the
    # keep>=1 floor still drafts one token per request (empty drafts crash the
    # verify path), so every request keeps exactly 1.
    surv = (0.6 * torch.ones(4, GAMMA)).cumprod(dim=1)
    keep = schedule_prefixes(surv, cost_a=0.001, cost_b=1000.0)
    assert torch.equal(keep, torch.ones(4, dtype=torch.int32))
