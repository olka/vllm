# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek V4 (arXiv 2606.19348).

Structure is pinned to the ``deepseek-ai/DeepSeek-V4-Flash-DSpark`` checkpoint
(``mtp.{0,1,2}.*`` -> ``model.layers.{43,44,45}.*``), confirmed against the model's
own ``inference/model.py`` reference (``DSparkBlock`` / ``forward_spec``):

  * ``mtp.0`` (first stage): ``main_norm`` + ``main_proj`` (fp8, ``[hidden, 3*hidden]``)
    that fuses the three target layers ``dspark_target_layer_ids = [40,41,42]`` into
    ``main_x``, plus a ``DeepseekV4DecoderLayer`` block. No heads, no ``hc_head``.
  * ``mtp.1`` (middle): a plain ``DeepseekV4DecoderLayer`` block.
  * ``mtp.2`` (head stage): block + ``norm`` + ``hc_head_{fn,base,scale}`` +
    ``markov_head`` (Eq. 5) + ``confidence_head`` (Eq. 7).

The draft owns no ``embed_tokens`` / LM head: both are shared from the target model
(MTP path in ``llm_base_proposer._maybe_share_*``). The Markov left-to-right block
sampling + confidence truncation live in the proposer (``vllm/v1/spec_decode/dspark.py``).

The DSpark *forward* (``forward_spec``) is a cross-attention pass — the noise-token block
is the query stream and ``main_x`` is the KV context (custom ``DSparkAttention`` in the
reference) — and is implemented in :meth:`DeepSeekV4DSparkModel.forward`. The attention
compute itself (Gap C) is being ported faithfully from the reference; see the
``NotImplementedError`` below for the exact algorithm.
"""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_dspark_heads import ConfidenceHead, MarkovHead
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.common.ops import (
    fused_q_kv_rmsnorm,
    mtp_shared_head_rmsnorm,
)
from vllm.sequence import IntermediateTensors

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)

# MoE expert scales: fp4 experts (Mxfp4MoEMethod) register ``w{1,2,3}_weight_scale``;
# fp8 block-quant experts register ``w{1,2,3}_weight_scale_inv``. All other fp8 linear
# scales (shared experts, main_proj, attn projections) use ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")

# Weights that live directly on the DSpark layer wrapper (everything else belongs to the
# wrapped ``DeepseekV4DecoderLayer`` and gets a ``.block.`` segment inserted on load).
_LAYER_LOCAL_NAMES = frozenset(
    {
        "main_norm",
        "main_proj",
        "norm",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "markov_head",
        "confidence_head",
    }
)


class DeepSeekV4DSparkLayer(nn.Module):
    """One DSpark draft stage: a V4 decoder block, plus the stage-0 target fusion
    (``main_norm``/``main_proj``) and the head-stage logits + sequential heads."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        *,
        is_first_layer: bool,
        is_head_layer: bool,
        topk_indices_buffer: torch.Tensor,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.is_first_layer = is_first_layer
        self.is_head_layer = is_head_layer
        quant_config = vllm_config.quant_config
        self.rms_norm_eps = config.rms_norm_eps
        hidden = config.hidden_size

        self.block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )

        # Stage 0 only: fuse the concatenated target layers into main_x.
        if is_first_layer:
            num_target = len(getattr(config, "dspark_target_layer_ids", []))
            self.main_norm = RMSNorm(hidden, eps=config.rms_norm_eps)
            self.main_proj = ReplicatedLinear(
                num_target * hidden,
                hidden,
                bias=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.main_proj",
            )

        # Head stage only: hypercompressed logits projection + the sequential heads.
        if is_head_layer:
            self.hc_eps = config.hc_eps
            self.hc_mult = config.hc_mult
            self.hc_dim = self.hc_mult * hidden
            self.norm = RMSNorm(hidden, eps=config.rms_norm_eps)
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32), requires_grad=False
            )
            self.markov_head = MarkovHead(
                config.vocab_size, hidden, config.dspark_markov_rank
            )
            self.confidence_head = ConfidenceHead(hidden, config.dspark_markov_rank)

    def fuse_target_hidden(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        """``main_norm(main_proj(concat of target layers))`` -> [num_tokens, hidden].

        main_proj (num_target*hidden -> hidden) runs first, then main_norm (sized
        hidden), matching the reference DSpark ``forward_embed``.
        """
        return self.main_norm(self.main_proj(aux_hidden))

    def project_logits_hidden(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Head stage: returns (pre_norm, post_norm) dense hidden [N, hidden].

        pre_norm = hc_head output (confidence head input, Phase 3); post_norm = +
        shared-head RMSNorm (LM-head input for base logits).
        """
        hidden_states = hidden_states.view(-1, self.hc_mult, self.config.hidden_size)
        # pre_norm: post-hc_head dense hidden [N, hidden] — fed to the confidence head
        # (Phase 3). post_norm: + shared-head RMSNorm — fed to the LM head for logits.
        pre_norm = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        post_norm = mtp_shared_head_rmsnorm(
            pre_norm.clone(),
            self.norm.weight.data,
            self.norm.variance_epsilon,
        )
        return pre_norm, post_norm


class DeepSeekV4DSparkModel(nn.Module):
    """Stacked DSpark draft (one stage per target layer) sharing the target embedding."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.start_layer_idx = config.num_hidden_layers
        # One draft stage per checkpoint MTP block (== len(target_layer_ids) == n_mtp=3
        # for V4-Flash-DSpark). HF ``num_nextn_predict_layers`` (=1) is unrelated here.
        self.num_draft_layers = len(getattr(config, "dspark_target_layer_ids", []))

        topk_tokens = config.index_topk
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
        )
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        last = self.start_layer_idx + self.num_draft_layers - 1
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekV4DSparkLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                    is_first_layer=(idx == self.start_layer_idx),
                    is_head_layer=(idx == last),
                    topk_indices_buffer=self.topk_indices_buffer,
                    aux_stream_list=aux_stream_list,
                )
                for idx in range(self.start_layer_idx, last + 1)
            }
        )
        # The draft shares the target model's embedding at runtime (MTP path).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    @property
    def head_layer(self) -> DeepSeekV4DSparkLayer:
        return self.layers[str(self.start_layer_idx + self.num_draft_layers - 1)]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        """EAGLE3/DFlash entry: fuse the concatenated target layers via stage 0."""
        return self.layers[str(self.start_layer_idx)].fuse_target_hidden(aux_hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the draft stages over the query block and return the pre-hc_head
        residual (``[num_tokens, hc_mult * hidden]``); ``compute_logits`` applies
        hc_head + norm + the shared LM head.

        Iteration 1 (runnable-lossless): the block self-attends through the SWA
        backend; the faithful ``main_x`` context cross-attention (precompute path)
        is layered in next, which is what lifts the acceptance rate.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # HC expansion to hc_mult copies, mirroring DeepseekV4Model.forward.
        x = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        input_ids = input_ids.to(torch.int64)

        residual = post_mix = res_mix = None
        for idx in range(
            self.start_layer_idx, self.start_layer_idx + self.num_draft_layers
        ):
            x, residual, post_mix, res_mix = self.layers[str(idx)].block(
                x, positions, input_ids, post_mix, res_mix, residual
            )
        x = mhc_post_tilelang(x, residual, post_mix, res_mix)
        return x.flatten(1)


class DeepSeekV4DSpark(nn.Module):
    """``ForCausalLM``-style wrapper exposing the proposer-facing API."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self._vllm_config = vllm_config
        self.model = DeepSeekV4DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Shared from the target model by the runner (MTP path); never has its own.
        self.lm_head: nn.Module | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden)

    @property
    def markov_head(self) -> MarkovHead:
        return self.model.head_layer.markov_head

    @property
    def confidence_head(self) -> ConfidenceHead:
        return self.model.head_layer.confidence_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Insert the cross-attention context K/V (from ``main_x``) into each draft
        stage's SWA cache before the query forward.

        ``context_states`` is the concatenation of the target layers [40,41,42]
        (``num_target*hidden``); fuse it once to ``main_x`` via the first stage, then
        for every draft stage project ``main_x`` -> KV (that stage's MLA attn) and
        insert it into the stage's SWA cache at ``context_slot_mapping``. Mirrors the
        per-stage KV side of ``_fused_qnorm_rope_kv_insert`` but with an explicit
        context slot mapping (no forward context here). Runs outside CUDA graph.
        """
        if context_positions.shape[0] == 0:
            return
        # Run the KV projection+insert inside set_forward_context so the fp8 MLA
        # linear (fused_wqa_wkv) takes the deep_gemm path correctly — the bare
        # hand-call outside any forward context tripped deep_gemm's shape assert.
        with set_forward_context(
            None, self._vllm_config, num_tokens=context_states.shape[0]
        ):
            self._insert_context_kv(
                context_states, context_positions, context_slot_mapping
            )

    def _insert_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None:
        if context_slot_mapping is None:
            return  # dummy_run/warmup: no cache to write, skip the context insert
        # Fuse the concatenated target layers to main_x once (stage 0's main_proj).
        first = self.model.layers[str(self.model.start_layer_idx)]
        num_target = len(getattr(self.config, "dspark_target_layer_ids", []))
        if context_states.shape[-1] == num_target * self.config.hidden_size:
            main_x = first.fuse_target_hidden(context_states)
        else:
            main_x = context_states
        # deep_gemm requires a contiguous fp8 input; main_x is a buffer slice / norm view.
        main_x = main_x.contiguous()

        for idx in range(
            self.model.start_layer_idx,
            self.model.start_layer_idx + self.model.num_draft_layers,
        ):
            attn = self.model.layers[str(idx)].block.attn
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            qr, kv = qr_kv.split([attn.q_lora_rank, attn.head_dim], dim=-1)
            # Match the real forward's pre-insert norm (kernel does not norm KV).
            qr, kv = fused_q_kv_rmsnorm(
                qr, kv, attn.q_norm.weight.data, attn.kv_norm.weight.data, attn.eps
            )
            if context_slot_mapping is None:
                continue  # dummy_run: exercise the projections, no cache write
            q = attn.wq_b(qr).view(-1, attn.n_local_heads, attn.head_dim)
            swa_cache = attn.swa_cache_layer.kv_cache
            cos_sin_cache = attn.rotary_emb.cos_sin_cache
            block_size = attn.swa_cache_layer.block_size
            if swa_cache.dtype == torch.uint8:  # fp8_ds_mla UE8M0 paged
                torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                    q,
                    kv,
                    swa_cache.view(swa_cache.shape[0], -1),
                    context_slot_mapping,
                    context_positions,
                    cos_sin_cache,
                    attn.padded_heads,
                    attn.eps,
                    block_size,
                )
            elif swa_cache.dtype == torch.bfloat16:
                torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                    q,
                    kv,
                    swa_cache.view(-1, block_size, attn.head_dim),
                    context_slot_mapping,
                    context_positions,
                    cos_sin_cache,
                    attn.eps,
                    block_size,
                )
            else:  # per-tensor float8_e4m3fn
                q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
                torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
                    q,
                    kv,
                    q_fp8,
                    swa_cache.view(-1, block_size, attn.head_dim),
                    context_slot_mapping,
                    context_positions,
                    cos_sin_cache,
                    attn._flashinfer_fp8_kv_scale,
                    attn._flashinfer_fp8_q_scale_inv,
                    attn.eps,
                    block_size,
                )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Base block logits ``U_k`` (Markov bias is added per-position in the proposer).

        Uses the head stage's hc_head + final norm, then the shared target LM head.
        """
        _, post_norm = self.model.head_layer.project_logits_hidden(hidden_states)
        return self.model.logits_processor(self.lm_head, post_norm)

    def compute_logits_and_conf_hidden(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (base_logits, pre_norm_hidden) — the proposer uses pre_norm as the
        confidence head's hidden input (Phase 3)."""
        pre_norm, post_norm = self.model.head_layer.project_logits_hidden(hidden_states)
        return self.model.logits_processor(self.lm_head, post_norm), pre_norm

    def _is_draft_layer(self, name: str) -> int | None:
        """Return the draft layer index encoded in ``name``, or None if out of range."""
        start = self.model.start_layer_idx
        for idx in range(start, start + self.model.num_draft_layers):
            if f"model.layers.{idx}." in name:
                return idx
        return None

    def _rewrite_layer_name(self, layer_idx: int, name: str) -> str:
        """Insert ``.block.`` for weights owned by the wrapped decoder layer; leave the
        DSpark-local weights (main_proj, norm, hc_head, markov/confidence) untouched."""
        prefix = f"model.layers.{layer_idx}."
        if not name.startswith(prefix):
            return name
        head = name[len(prefix) :].split(".", 1)[0]
        if head in _LAYER_LOCAL_NAMES:
            return name
        return prefix + "block." + name[len(prefix) :]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the bundled DSpark draft tensors from ``mtp.{i}.*``.

        Ported from ``nvidia/mtp.py:load_weights`` (``.mtp_block`` -> ``.block``), with
        the DSpark-only ``main_proj``/``norm``/``hc_head``/``markov``/``confidence``
        weights routed directly to the layer wrapper.
        """

        def _find_mtp_idx(name: str) -> int:
            for sub in name.split("."):
                if sub.isdigit():
                    return int(sub)
            return 0

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        first_layer = next(iter(self.model.layers.values()))
        if first_layer.block.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )

        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        nhl = self.config.num_hidden_layers
        for name, loaded_weight in weights:
            mtp_idx = _find_mtp_idx(name)
            name = name.replace(f"mtp.{mtp_idx}.", f"model.layers.{nhl + mtp_idx}.")

            layer_idx = self._is_draft_layer(name)
            if layer_idx is None:
                continue
            name = self._rewrite_layer_name(layer_idx, name)
            # markov_w{1,2} are bare nn.Parameters here (no ``.weight`` child).
            if ".markov_head.markov_w" in name and name.endswith(".weight"):
                name = name.removesuffix(".weight")

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Stacked params (gate_up, fused_wqa_wkv) live inside the decoder
                # block; the ``.block.`` guard avoids ``w1`` matching ``markov_w1``.
                if ".block." not in name:
                    continue
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            loaded_params.add(name_mapped)
                            break
                    continue
                elif "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if ".shared_experts.w2" in name:
                        name = name.replace(
                            ".shared_experts.w2", ".shared_experts.down_proj"
                        )
                    if name.endswith(".ffn.gate.bias"):
                        name = name.replace(
                            ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                        )
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        loaded_layers = {self._is_draft_layer(p) for p in loaded_params}
        for layer_idx in range(
            self.model.start_layer_idx,
            self.model.start_layer_idx + self.model.num_draft_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"DSpark draft layer {layer_idx} weights missing from the "
                    f"checkpoint. Use a checkpoint that bundles the DSpark drafter."
                )
        self.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.model.layers.values():
            layer.block.ffn.finalize_mega_moe_weights()
