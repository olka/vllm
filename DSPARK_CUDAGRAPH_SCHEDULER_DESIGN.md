# DSpark cudagraph-native confidence scheduler — design

## Thesis
The DSpark confidence head + hardware-aware router is the mechanism that **unblocks
draft length > 3**: it trims the verification tail so long drafts stop being eaten by
verify overhead. Its highest-value target is **agentic / structured-output** serving,
where token predictability has high *intra-sequence variance* (rigid JSON/tool-call
scaffolding interleaved with novel values) — exactly what a fixed γ can't exploit and
an adaptive router can. The blocker to shipping it on NVIDIA was CUDA-graph
incompatibility. This design makes it **graph-native**.

## Why it looked graph-incompatible (the naive framing)
CUDA graphs freeze **shapes + addresses** at capture; replay requires matching shapes.
A *per-request* verify length `keep_r` makes the total query length
`s_q = Σ_r (1 + keep_r)` **data-dependent**, so no captured shape matches and the
sparse-MLA attention metadata (`indices` of shape `(s_q, h_kv, topk)`) mismatches →
the `flash_mla_cuda.sparse_prefill_fwd: indices must have shape` crash we hit at N=300.

Two non-answers:
- **Pad each request to γ, mask the rest (Flavor A):** fixed shape, but you verify the
  full γ for everyone → the trim is computed and discarded → **no saving** (≈ no scheduler).
- **Per-request pack+bucket (Flavor B):** real saving but needs varlen-packed verify +
  sparse-MLA metadata surgery. Deferred to v2.

## Key insight (the unlock)
Graphs fix **shape**, not **contents** — vLLM already refills fixed-address input
buffers each step and replays. So make the scheduler's decision a **single per-step
integer `L`** (the batch-uniform verify length) instead of a per-request vector. Then:

- shape = `num_reqs · (1 + L)` → **fixed for a given (num_reqs, L)** → graph-capturable.
- `L` simply **selects which pre-captured graph to replay.**
- `L ∈ {1..n}`, `n` from config (block/max-draft length) → a **finite, init-known** set.

The thing we thought was fundamentally graph-incompatible becomes graph-*native* the
moment the decision is one bounded per-step integer.

## Design

### Init: capture + calibrate is ONE loop
Walk the grid `{num_reqs buckets} × {L = 1..n}`. For each cell:
1. Capture the decode/verify cudagraph for shape `num_reqs · (1+L)`.
2. **Time it** → record `SPS(B, L)` (tokens/s).

Output: (a) the replayable graphs, (b) the `SPS(B, L)` table. The paper's hardware-aware
calibration (`Θ = τ·SPS(B)`) and vLLM's cudagraph capture are the **same init pass** —
timing each captured graph *is* the throughput profile the router needs.

### Runtime (fully graphed)
1. Draft the full block — fixed shape, graphed.
2. Confidence head → per-request survival probabilities (informs, not wasted).
3. Router: `Θ = τ·SPS(B)` → choose `L ∈ [1,n]` maximizing expected
   accepted-tokens-per-cost given the survival distribution + the `SPS` table.
4. **Replay `graph[B, L]`.** Rejection-sample, accept prefix.

The router's entire runtime job is **pick the index `L`**. No ragged tensors, no packing,
no per-step metadata build.

### L-selection rule (v1)
Given per-request survival probs `p_r(j)` (prob the prefix survives to draft position j)
and the SPS table, pick:
`L* = argmax_L [ Σ_r Σ_{j≤L} p_r(j) ] · SPS(B, L)`  (expected accepted tokens × throughput),
i.e. the batch-level verify length where marginal expected-accept stops paying for the
marginal verify cost. (Exact objective TBD; start simple — survival-quantile threshold.)

## v1 vs v2
- **v1 (this doc): per-step uniform `L`.** Graph-native, trivial plumbing, captures the
  bulk of the value. Granularity loss is small in **agentic batches** (requests are
  correlated — many in tool-call mode together), and it still **adapts over time** as the
  batch moves through scaffolding ↔ content.
- **v2: per-request pack+bucket.** Only if heterogeneous-batch granularity proves to matter.

## Why it matters (positioning)
- Unblocks γ>3 — long-horizon speculation — which is where agentic/structured workloads live.
- Agentic serving is mainstream NVIDIA/vLLM territory (high batch + structurally
  predictable) → not an HAI-LLM-only corner.
- Constrained decoding (JSON mode / grammar) amplifies it (forced scaffolding → ~100%
  draft accept on those tokens), and pairs well with the cheap Markov head.
- None of this exists in 00b1798 (confidence head loaded but `predict_confidence` has zero
  callers; no scheduler) or any upstream PR. It's a **follow-up that builds on their base**,
  not a competitor — to land when base support merges.

## Benchmark plan
- Wikitext understates it (uniform predictability). Use **tool-call / structured-output /
  agentic traces**, ideally with grammar constraints.
- Compare at **long γ**: (A) eager no-scheduler, (C) eager scheduler, (B) cudagraph
  no-scheduler, and the new **(D) cudagraph + per-step-L scheduler**. Target: D > B and
  D > short-γ baseline.

## Dev / PR strategy
- Develop + measure on our **V1 branch** (it runs on B300; 00b1798's V2 doesn't —
  non-causal draft + attention sinks → FlashInfer crash).
- Keep rebased against their evolving base; submit as a follow-up PR when base merges.
- Disclosure: AI-assisted; human owns/defends; run full v1 test suite before submit.

## Implementation plan (files) — filled after code recon

**Key recon finding:** vLLM graphs are keyed by **`num_tokens` (total)**, not per-request.
- **FULL** mode bakes in `(num_tokens, num_reqs, uniform)` and derives num_reqs assuming
  fixed query_len `1+γ` → adding `L` here = capture per `(num_tokens, L)` = L-fold memory.
  We DON'T use FULL (it corrupts DSpark drafts → 0.5% accept).
- **PIECEWISE** (what we use) relaxes the key to **`num_tokens` only** (num_reqs=None),
  captures per-token compute (FFN/MoE), leaves attention eager.

**Therefore the uniform-L design needs NO capture changes.** In PIECEWISE, uniform `L`
makes `num_tokens = num_reqs·(1+L)` — already a bucketed key → smaller L → smaller
existing graph → automatic saving. And the SPS table is unnecessary for v1: the dispatcher
already rounds num_tokens to a captured bucket; our online `cost_a/cost_b` model picks L.

**Why our current ragged version is eager-only (recon):** returning a ragged
`list[list[int]]` routes the runner to a NON-cudagraph path
(`gpu_model_runner.py` `_get_draft_token_ids_cpu` returns the list as-is;
`_copy_draft_token_ids_to_cpu` skips it). Returning a uniform tensor keeps the cudagraph path.

**v1 changes (mostly our own file):**
- [ ] `schedule_prefixes`/`_schedule_keep` → return a **scalar L** (reuse `cost_a/cost_b`)
      — `vllm/v1/spec_decode/dspark.py`
- [ ] `propose` → return uniform **`[num_reqs, L]`** tensor (drop ragged `[row[:k]]`)
      — `vllm/v1/spec_decode/dspark.py`
- [ ] confirm decode routing accepts `query_len = 1+L < 1+γ` (our `decode_threshold`
      fix already targets this) — `vllm/v1/attention/backends/mla/sparse_swa.py`
- [ ] **VALIDATE FIRST (cheap, decides approach):** does dropping L land in a *smaller*
      PIECEWISE `num_tokens` bucket (real saving, not rounded back up), and is the
      sparse-MLA verify correct for uniform `L<γ`? Test on V1 before building the L-rule.

Recon refs: graph key `forward_context.py:37` (BatchDescriptor.num_tokens);
PIECEWISE relax `cudagraph_dispatcher.py:313-318`; FULL num_reqs derive `:144`;
verify metadata `gpu_model_runner.py:2761-2834` (`_calc_spec_decode_metadata`);
ragged non-graph path `gpu_model_runner.py:4798-4799`.

**Capture-dimension `L` + SPS-timed init loop = only needed for a FULL-mode v3.**
