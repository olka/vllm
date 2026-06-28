# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative-decoding proposer (arXiv 2606.19348).

DSpark = DFlash parallel block backbone + a low-rank Markov sequential head + a
confidence head. We subclass :class:`DFlashProposer` to inherit the parallel
mask-token-block machinery (cross-attention input layout, the fused input Triton
kernel, CUDA-graph-stable buffers, ``dummy_run``) and override only the
*sampling tail*:

  1. one parallel backbone pass -> base logits ``U_1..U_gamma`` + hiddens ``h_1..h_gamma``;
  2. a left-to-right **Markov loop** that adds ``B(x_{k-1}) = W1[x_{k-1}] @ W2.T`` before
     sampling each position (injects the intra-block dependency a pure parallel drafter
     lacks). This touches only logits — no transformer re-run — so it stays in the
     ``T_sequential << T_parallel`` regime, and is a fixed-length (``gamma``) unroll that
     is CUDA-graph-capturable;
  3. **confidence-head truncation**: keep the longest prefix whose cumulative survival
     probability stays above ``confidence_threshold`` (static-threshold variant; the
     paper's hardware-aware scheduler is out of scope).

The Markov-*adjusted* logits — not the base logits — are returned as the draft
distribution so that rejection-sampling verification stays exact (lossless).
"""

import time

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


def fit_linear_cost(
    samples: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Least-squares fit of step-time ``T(B) = a + b*B`` over (budget, time) samples.

    Returns ``(a, b)`` with ``a > 0`` and ``b > 0``, or ``None`` when there is no
    budget variation yet (a degenerate fit, e.g. only one batch size seen).
    """
    n = len(samples)
    if n < 2:
        return None
    s_b = sum(b for b, _ in samples)
    s_t = sum(t for _, t in samples)
    s_bb = sum(b * b for b, _ in samples)
    s_bt = sum(b * t for b, t in samples)
    denom = n * s_bb - s_b * s_b
    if denom <= 0.0:
        return None
    b = (n * s_bt - s_b * s_t) / denom
    a = (s_t - b * s_b) / n
    if a <= 0.0 or b <= 0.0:
        return None
    return a, b


def schedule_prefixes(
    surv: torch.Tensor, cost_a: float | None, cost_b: float | None
) -> torch.Tensor:
    """Hardware-aware prefix scheduler (paper §3.2.2, Algorithm 1).

    Given cumulative prefix-survival probs ``surv`` ``[R, gamma]`` (monotone
    non-increasing in position) and the cost model ``T(B) = cost_a + cost_b*B``,
    pick the verification budget maximizing accepted-tokens / step-time and return
    per-request keep lengths ``[R]`` (int32). Because ``a_{r,j}`` is monotone, the
    top-k survival probs across the batch form valid prefixes. Falls back to the
    full block (``gamma``) when the cost model is uncalibrated (``cost_b`` falsy).
    """
    R, gamma = surv.shape
    if not cost_b:
        return surv.new_full((R,), gamma, dtype=torch.int32)
    sorted_a, order = surv.reshape(-1).sort(descending=True)
    tau = torch.cat([surv.new_zeros(1), sorted_a.cumsum(0)])  # [R*gamma+1]
    k = torch.arange(R * gamma + 1, device=surv.device, dtype=surv.dtype)
    budget = R + k  # total verify tokens = R bonus + k draft positions
    throughput = (R + tau) / (cost_a + cost_b * budget)
    kstar = int(throughput.argmax())
    keep = surv.new_zeros(R, dtype=torch.int32)
    if kstar:
        req = (order[:kstar] // gamma).to(torch.int64)
        keep.scatter_add_(
            0, req, torch.ones(kstar, dtype=torch.int32, device=surv.device)
        )
    return keep


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        # DSpark IS a DFlash-style parallel drafter (arXiv 2606.19348), so reuse
        # DFlashProposer's setup verbatim — its assert accepts the dspark method.
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)

        # DSpark's draft block is semi-autoregressive: non-causal block attention
        # (every block query attends to the full window incl. all block positions),
        # with the sequential dependency carried by the Markov head. The SWA backend
        # honors this via the metadata `causal=False` flag (set from this).
        self.dflash_causal = False
        # DSpark samples from the anchor at query position 0 (its real-token hidden
        # predicts the first draft token, per the reference forward_head) — unlike
        # DFlash, which skips the bonus and samples only the mask positions.
        self._sample_from_bonus = True

        cfg = self._dspark_config
        self.block_size = cfg.get("block_size", self.num_speculative_tokens)
        self.markov_rank = cfg.get("markov_rank", 256)
        # Confidence-scheduled verification (paper §3.2): the confidence head scores
        # per-position prefix survival, and the hardware-aware prefix scheduler
        # (§3.2.2) allocates the per-batch verification budget to maximize accepted
        # tokens per step-time. Off by default; the scheduler only prunes when the
        # online-calibrated cost model shows verification is compute-bound, so it
        # preserves the full block (and the latency win) at small batch.
        self.enable_confidence = cfg.get("enable_confidence_head", False)
        # Online-calibrated verify cost model T(B) = cost_a + cost_b * B, fit from
        # observed (verify_budget, step_time) samples. None until enough variation.
        self._cost_a: float | None = None
        self._cost_b: float | None = None
        self._cost_samples: list[tuple[float, float]] = []
        self._last_step_time: float | None = None
        self._last_budget: float | None = None
        self._last_surv: torch.Tensor | None = None

    @property
    def _dspark_config(self) -> dict:
        return getattr(self.draft_model_config.hf_config, "dspark_config", None) or {}

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        # DSpark fuses target layers [40,41,42] into main_x, so request aux hidden
        # states from the target (which now implements the EAGLE3 interface).
        return True

    # ------------------------------------------------------------------
    # The one new piece: sequential Markov refinement + confidence truncation.
    # ------------------------------------------------------------------
    def _markov_block_sample(
        self,
        base_logits: torch.Tensor,  # [batch, gamma, vocab]
        hidden: torch.Tensor,  # [batch, gamma, hidden]
        anchor_token_ids: torch.Tensor,  # [batch] — bonus token x_0 from this round
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Batched left-to-right Markov sampling + confidence scoring.

        Returns:
            draft_token_ids: [batch, gamma]
            draft_logits:    [batch, gamma, vocab]  (Markov-adjusted; the draft
                             distribution handed to the rejection sampler)
            surv:            [batch, gamma] cumulative prefix-survival probs, or None
                             if the confidence head is disabled
        """
        markov = self.model.markov_head
        conf = self.model.confidence_head if self.enable_confidence else None
        batch, gamma, _ = base_logits.shape

        prev = anchor_token_ids  # [batch]
        tokens: list[torch.Tensor] = []
        adjusted: list[torch.Tensor] = []
        conf_logits: list[torch.Tensor] = []
        for k in range(gamma):
            w1_prev = markov.prev_token_embedding(prev)  # [batch, rank]
            bias = markov.bias_from_embedding(w1_prev)  # [batch, vocab]
            logit_k = base_logits[:, k, :] + bias
            adjusted.append(logit_k)
            # Greedy keeps verification exact for greedy requests; multinomial path is
            # selected per-request by sampling_metadata in the full integration.
            x_k = logit_k.argmax(dim=-1)  # [batch]
            tokens.append(x_k)
            if conf is not None:
                conf_logits.append(conf.logits(hidden[:, k, :], w1_prev))  # [batch]
            prev = x_k

        draft_token_ids = torch.stack(tokens, dim=1)  # [batch, gamma]
        draft_logits = torch.stack(adjusted, dim=1)  # [batch, gamma, vocab]

        if conf is None:
            return draft_token_ids, draft_logits, None

        # Per-position cumulative prefix-survival probability a_{r,j} (monotone in j).
        # The hardware-aware scheduler consumes this to choose the verify budget.
        c = torch.stack(conf_logits, dim=1).sigmoid()  # [batch, gamma]
        surv = c.float().cumprod(dim=1)  # [batch, gamma]
        return draft_token_ids, draft_logits, surv

    # ------------------------------------------------------------------
    # Hardware-aware prefix scheduler (paper §3.2.2, Algorithm 1).
    # ------------------------------------------------------------------
    def _record_cost(self, budget: float, dt: float) -> None:
        """Accumulate a (verify_budget, step_time) sample and refit the cost model."""
        if dt <= 0.0 or dt > 5.0:  # drop prefill spikes / scheduler stalls
            return
        self._cost_samples.append((budget, dt))
        if len(self._cost_samples) > 256:
            self._cost_samples.pop(0)
        if len(self._cost_samples) >= 16:
            self._fit_cost()

    def _fit_cost(self) -> None:
        """Refit the cost model from samples (natural batch drain supplies the B
        variation needed to fit the slope during keep-all bootstrap)."""
        fit = fit_linear_cost(self._cost_samples)
        if fit is not None:
            self._cost_a, self._cost_b = fit

    def _schedule_keep(self, surv: torch.Tensor) -> torch.Tensor:
        return schedule_prefixes(surv, self._cost_a, self._cost_b)

    @override
    def propose(self, *args, **kwargs) -> torch.Tensor | list[list[int]]:
        """Run the inherited DFlash parallel pass; the Markov refinement is spliced in
        via the overridden :meth:`_sample_draft_tokens`. We stash the anchor token
        x_0, time the step for cost-model calibration, and apply the hardware-aware
        prefix scheduler to emit a ragged (per-request length) draft.
        """
        # next_token_ids is the 5th positional arg of the base propose().
        self._anchor_token_ids = (
            args[4] if len(args) > 4 else kwargs["next_token_ids"]
        )
        now = time.perf_counter()
        if self._last_step_time is not None and self._last_budget is not None:
            self._record_cost(self._last_budget, now - self._last_step_time)
        self._last_step_time = now

        draft = super().propose(*args, **kwargs)  # [num_reqs, gamma] tensor
        if not self.enable_confidence or self._last_surv is None:
            self._last_budget = None
            return draft
        keep = self._schedule_keep(self._last_surv)
        self._last_budget = float(self._last_surv.shape[0] + int(keep.sum()))
        # Ragged draft -> runner's variable-length path (like ngram): the engine
        # verifies only `keep` tokens per request, so the drafted denominator drops.
        return [row[:k] for row, k in zip(draft.tolist(), keep.tolist())]

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Markov-refined block sampling over the parallel backbone's base logits.

        ``hidden_states`` holds the gamma sample positions per request (request-major,
        ``[batch*gamma, *]``). We compute the base logits ``U_k``, then add the Markov
        bias ``B(x_{k-1})`` left-to-right (Eq. 4-5). Greedy sampling -> draft_probs=None,
        which gives exact (lossless) rejection-sampling acceptance.
        """
        gamma = self.num_speculative_tokens
        if self.enable_confidence:
            # Confidence needs the post-hc_head dense hidden [N, hidden], not the flat
            # hc_mult*hidden residual.
            base_logits, conf_hidden = self.model.compute_logits_and_conf_hidden(
                hidden_states
            )
        else:
            base_logits = self.model.compute_logits(hidden_states)
            conf_hidden = hidden_states
        vocab = base_logits.shape[-1]
        base_logits = base_logits.view(-1, gamma, vocab)
        hidden = conf_hidden.view(-1, gamma, conf_hidden.shape[-1])
        draft_token_ids, _draft_logits, surv = self._markov_block_sample(
            base_logits, hidden, self._anchor_token_ids
        )
        # `surv` (cumulative prefix-survival probs) is consumed by the hardware-aware
        # scheduler in propose(); None when the confidence head is disabled.
        self._last_surv = surv
        return draft_token_ids.view(-1), None
