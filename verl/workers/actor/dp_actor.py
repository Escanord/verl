# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_global_entropy_top_mask, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.walk_importance import WalkImportanceComputer, TemporalWalkComputer, RepresentationDivergenceComputer
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        self._pivot_update_step = 0  # incremented once per update_policy call; used for internalize decay

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            _lan_grpo_log_p = None      # PIVOT: set in rmpad path when lan_grpo_coeff > 0
            _lan_grpo_mask = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                _dv_stats: dict = {}

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # PIVOT-v6: Langevin internalization KL loss.
                    # Computed entirely from this micro-batch's logits — no rollout-time
                    # data needed.  For each prompt group of n consecutive rollouts:
                    #   1. Compute group-mean logits and ΔVar at each response step.
                    #   2. Trigger positions: ΔVar > delta_var_threshold (same rule as rollout).
                    #   3. p_langevin = softmax((1-η)*logits + η*group_mean)  [deterministic].
                    #   4. L = mean_{b,t:trigger, adv>0} [adv * KL(p_lan || π_θ, top-K)].
                    # Teacher-forced logits == autoregressive logits for causal transformers,
                    # so trigger positions and p_langevin here match those from rollout exactly.
                    _pivot_cfg = getattr(self.config, "pivot", None) or {}
                    _lan_grpo_coeff = float(_pivot_cfg.get("lan_grpo_coeff", 0.0))
                    _correct_is = bool(_pivot_cfg.get("lan_grpo_correct_is", False))
                    _correct_mu = bool(_pivot_cfg.get("lan_grpo_correct_mu", False))
                    # pivot_version controls the IS denominator formula:
                    #   2 → entropy Langevin: softmax(logits + η*∇H(logits))
                    #   4 → group mean pull:  (1-η)*π + η*softmax(μ_group)
                    # Must match the pivot_version used in the rollout.
                    _pivot_version_actor = int(_pivot_cfg.get("pivot_version", 4))
                    _compute_internalize_kl = (
                        _lan_grpo_coeff > 0.0
                        and not is_mask_all_zero
                        and not self.use_ulysses_sp
                    )
                    _lan_grpo_log_p = None
                    _lan_grpo_mask = None

                    if _compute_internalize_kl:
                        _n = int(_pivot_cfg.get("n_rollouts_per_prompt", 8))
                        _trig_percentile = _pivot_cfg.get("trig_percentile", None)
                        _fixed_dv_thresh = float(_pivot_cfg.get("delta_var_threshold", 5.0))
                        if _trig_percentile is not None:
                            _actor_dv_buf = getattr(self, "_actor_pivot_dv_buf", [])
                            if len(_actor_dv_buf) >= 200:
                                _buf_sorted = sorted(_actor_dv_buf[-2000:])
                                _idx = int(len(_buf_sorted) * float(_trig_percentile) / 100.0)
                                _idx = max(0, min(_idx, len(_buf_sorted) - 1))
                                _dv_thresh = _buf_sorted[_idx]
                            else:
                                _dv_thresh = _fixed_dv_thresh
                        else:
                            _dv_thresh = _fixed_dv_thresh
                        _entropy_trigger_only = bool(_pivot_cfg.get("entropy_trigger_only", False))
                        _ent_thresh = float(_pivot_cfg.get("entropy_threshold", 8.0))
                        # Adaptive entropy threshold: mirrors trig_percentile for ΔVar path.
                        # Uses historical mean_ent buffer so trigger rate stays at (100-p)%.
                        if _entropy_trigger_only and _trig_percentile is not None:
                            _actor_ent_buf = getattr(self, "_actor_pivot_ent_buf", [])
                            if len(_actor_ent_buf) >= 200:
                                _ent_buf_sorted = sorted(_actor_ent_buf[-2000:])
                                _ent_idx = int(len(_ent_buf_sorted) * float(_trig_percentile) / 100.0)
                                _ent_idx = max(0, min(_ent_idx, len(_ent_buf_sorted) - 1))
                                _ent_thresh = _ent_buf_sorted[_ent_idx]
                        _eta = float(_pivot_cfg.get("langevin_eta", 0.1))
                        _top_k = int(_pivot_cfg.get("langevin_top_k", 20))
                        _adv = micro_batch.get("advantages")       # (B, response_length)
                        _rmask = micro_batch.get("response_mask")  # (B, response_length)
                        _responses = micro_batch.get("responses")  # (B, response_length)
                        # Rollout trigger masks from PIVOTv2LangevinAdapter._TRIGGER_REGISTRY.
                        # numpy object array of shape (B,); each element is list[float] or None.
                        # Value > 0 at response position t means Langevin fired there during rollout.
                        _rollout_trigger_raw = micro_batch.get("lan_trigger_mask", None)
                        # Pre-computed IS denominator log(p_lan(x_t)) from rollout.
                        # Shape: (B,) object array; each element list[float] or None.
                        # When present, skips the expensive (n, n_trig, vocab) logit tensor.
                        _rollout_log_p_lan = micro_batch.get("lan_log_p_lan", None)
                        if _lan_grpo_coeff > 0.0 and _rollout_log_p_lan is None and _rollout_trigger_raw is not None:
                            raise RuntimeError(
                                "lan_grpo_coeff > 0 but lan_log_p_lan missing from batch — "
                                "check that data.select() includes 'lan_log_p_lan' in "
                                "non_tensor_select_keys (dp_actor.py update_actor)."
                            )

                        _prereqs = (
                            _adv is not None
                            and _rmask is not None
                            and batch_size % _n == 0
                        )

                        if _prereqs:
                            _n_groups = batch_size // _n
                            _resp_lens = _rmask.sum(dim=-1).long()          # (B,)
                            _total_lens = attention_mask.sum(dim=-1).long()  # (B,)
                            _prompt_lens = _total_lens - _resp_lens          # (B,)
                            _resp_starts = cu_seqlens[:batch_size] + _prompt_lens  # (B,)

                            _can_lan_grpo = _lan_grpo_coeff > 0.0 and _responses is not None
                            # Per-group tensors accumulated across groups; cat'd after the loop.
                            _lan_grpo_b_idx: list[torch.Tensor] = []
                            _lan_grpo_t_idx: list[torch.Tensor] = []
                            _lan_grpo_vals: list[torch.Tensor] = []
                            _all_grp_var: list[torch.Tensor] = []  # for delta_var calibration logging
                            _all_grp_ent: list[torch.Tensor] = []  # for entropy calibration logging
                            for _g in range(_n_groups):
                                _s, _e = _g * _n, (_g + 1) * _n
                                _min_resp = int(_resp_lens[_s:_e].min().item())
                                if _min_resp == 0:
                                    continue

                                _resp_starts_g = [int(_resp_starts[_b]) for _b in range(_s, _e)]
                                _w_mean: torch.Tensor = None  # type: ignore[assignment]

                                if _rollout_trigger_raw is not None and _entropy_trigger_only:
                                    # Trigger positions already known from rollout — skip Welford entirely.
                                    # Saves (n, min_resp, vocab) bf16 tensors + entropy softmax passes.
                                    # Use _max_resp (not _min_resp) so a single short rollout in the group
                                    # doesn't suppress triggers from the other long rollouts.
                                    # _valid_lg downstream already enforces per-rollout length bounds.
                                    _max_resp = int(_resp_lens[_s:_e].max().item())
                                    _grp_union_trig = torch.zeros(
                                        _max_resp, dtype=torch.bool, device=logits_rmpad.device
                                    )
                                    for _b_g in range(_s, _e):
                                        _rm = _rollout_trigger_raw[_b_g]
                                        if _rm is not None and len(_rm) > 0:
                                            _rm_len = min(len(_rm), _max_resp)
                                            _rm_t = torch.tensor(_rm[:_rm_len], dtype=torch.float32, device=logits_rmpad.device)
                                            _grp_union_trig[:_rm_len] |= (_rm_t > 0)
                                    _trig = _grp_union_trig.nonzero(as_tuple=True)[0]
                                else:
                                    # Pass 1: streaming Welford variance — one (min_resp, vocab)
                                    # slice at a time.  Avoids materializing (n, min_resp, vocab)
                                    # which OOMs for long responses (vocab=151936, n=8, seq=4096).
                                    # torch.no_grad(): trigger positions are non-differentiable;
                                    # without this, 8 × (min_resp, vocab) float32 _x/_delta tensors
                                    # accumulate in the autograd graph across micro-batches.
                                    _w_M2: torch.Tensor = None    # type: ignore[assignment]
                                    _w_ent_sum: torch.Tensor = None  # type: ignore[assignment]
                                    _w_count = 0
                                    with torch.no_grad():
                                        for _rs in _resp_starts_g:
                                            _x = logits_rmpad[_rs:_rs + _min_resp]  # keep in bf16 to halve Welford memory
                                            _x = _x.where(torch.isfinite(_x), torch.zeros_like(_x))
                                            _w_count += 1
                                            if _w_mean is None:
                                                _w_mean = _x.clone()
                                                _w_M2 = torch.zeros_like(_x)
                                            else:
                                                _delta = _x - _w_mean
                                                _w_mean.add_(_delta / _w_count)
                                                _w_M2.add_(_delta * (_x - _w_mean))
                                            if _entropy_trigger_only:
                                                _p_x = torch.softmax(_x.float(), dim=-1)
                                                _ent_x = -(_p_x * _p_x.clamp(min=1e-12).log()).sum(-1)  # (min_resp,)
                                                _w_ent_sum = _ent_x if _w_ent_sum is None else _w_ent_sum + _ent_x
                                        # _grp_var[t] = pop-var over rollouts averaged over vocab
                                        _grp_var = (_w_M2 / _w_count).mean(dim=-1)  # (min_resp,)
                                    del _w_M2  # free before pass 2
                                    _all_grp_var.append(_grp_var.detach().float().cpu())
                                    if _entropy_trigger_only:
                                        _mean_ent = _w_ent_sum / _w_count  # (min_resp,)
                                        _all_grp_ent.append(_mean_ent.detach().float().cpu())
                                        _trig = (_mean_ent > _ent_thresh).nonzero(as_tuple=True)[0]
                                    else:
                                        _trig = (_grp_var > _dv_thresh).nonzero(as_tuple=True)[0]
                                if _trig.numel() == 0:
                                    del _w_mean
                                    continue

                                # When rollout pre-computed log(p_lan(x_t)) is present we read
                                # the IS denominator directly — no (n, n_trig, vocab) tensor needed.
                                # If lan_log_p_lan is absent (data.select() bug or no Langevin data)
                                # we skip IS correction; the assertion above would have already
                                # raised if lan_grpo_coeff > 0 and the key is missing.
                                _need_grp_trig = _can_lan_grpo and _rollout_log_p_lan is None
                                if not _need_grp_trig:
                                    if _can_lan_grpo:
                                        _n_trig = _trig.numel()
                                        _trig_list = _trig.tolist()
                                        _log_p_lan_tok_rows = []
                                        for _b_rel_lp in range(_n):
                                            _b_g_lp = _s + _b_rel_lp
                                            _lp_row = _rollout_log_p_lan[_b_g_lp]
                                            _log_p_lan_tok_rows.append([
                                                float(_lp_row[_ti])
                                                if _lp_row is not None and _ti < len(_lp_row)
                                                else 0.0
                                                for _ti in _trig_list
                                            ])
                                        _log_p_lan_tok = torch.tensor(
                                            _log_p_lan_tok_rows,
                                            dtype=torch.float32, device=logits_rmpad.device,
                                        )  # (n, n_trig) — frozen from rollout, no grad
                                        _valid_lg = _trig.unsqueeze(0) < _resp_lens[_s:_e].unsqueeze(1)
                                        if _rollout_trigger_raw is not None and _entropy_trigger_only:
                                            _per_rollout_valid = torch.zeros(
                                                _n, _n_trig, dtype=torch.bool, device=_trig.device
                                            )
                                            for _b_rel_lg, _b_g_lg in enumerate(range(_s, _e)):
                                                _rm_lg = _rollout_trigger_raw[_b_g_lg]
                                                if _rm_lg is not None:
                                                    for _ki, _ti in enumerate(_trig_list):
                                                        if _ti < len(_rm_lg) and _rm_lg[_ti] > 0:
                                                            _per_rollout_valid[_b_rel_lg, _ki] = True
                                            _valid_lg = _valid_lg & _per_rollout_valid
                                        if _valid_lg.any():
                                            _b_rel_v, _k_v = _valid_lg.nonzero(as_tuple=True)
                                            _lan_grpo_b_idx.append(_b_rel_v + _s)
                                            _lan_grpo_t_idx.append(_trig[_k_v])
                                            _lan_grpo_vals.append(_log_p_lan_tok[_b_rel_v, _k_v])
                                    del _w_mean
                                    continue
                            if _all_grp_var:
                                _dv_cat = torch.cat(_all_grp_var)
                                _dv_ps = torch.quantile(_dv_cat, torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99]))
                                _dv_stats = {
                                    "p50": _dv_ps[0].item(), "p75": _dv_ps[1].item(),
                                    "p90": _dv_ps[2].item(), "p95": _dv_ps[3].item(),
                                    "p99": _dv_ps[4].item(), "max": _dv_cat.max().item(),
                                }
                                # Update persistent buffer for adaptive threshold.
                                if _trig_percentile is not None:
                                    _buf = getattr(self, "_actor_pivot_dv_buf", [])
                                    _buf.extend(_dv_cat.tolist())
                                    if len(_buf) > 4000:
                                        _buf = _buf[-2000:]
                                    self._actor_pivot_dv_buf = _buf
                            if _entropy_trigger_only and _trig_percentile is not None and _all_grp_ent:
                                _ent_cat = torch.cat(_all_grp_ent)
                                _ent_buf = getattr(self, "_actor_pivot_ent_buf", [])
                                _ent_buf.extend(_ent_cat.tolist())
                                if len(_ent_buf) > 4000:
                                    _ent_buf = _ent_buf[-2000:]
                                self._actor_pivot_ent_buf = _ent_buf

                            if _lan_grpo_vals:
                                _b_t = torch.cat(_lan_grpo_b_idx)
                                _t_t = torch.cat(_lan_grpo_t_idx)
                                _vals_t = torch.cat(_lan_grpo_vals)
                                _lan_grpo_log_p = _adv.new_zeros(_adv.shape).index_put(
                                    (_b_t, _t_t), _vals_t
                                )  # (B, resp_len), log p_lan_current at trigger pos, 0 elsewhere
                                _lan_grpo_mask = torch.zeros(
                                    _adv.shape, dtype=torch.bool, device=_adv.device
                                ).index_put(
                                    (_b_t, _t_t),
                                    torch.ones(_vals_t.numel(), dtype=torch.bool, device=_adv.device)
                                )  # (B, resp_len), True at trigger positions

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = not calculate_entropy
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if _lan_grpo_log_p is not None:
                outputs["lan_grpo_log_p"] = _lan_grpo_log_p
                outputs["lan_grpo_mask"] = _lan_grpo_mask
            if _dv_stats:
                outputs["dv_stats"] = _dv_stats
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")
        if "lan_trigger_mask" in data.non_tensor_batch:
            non_tensor_select_keys.append("lan_trigger_mask")
        if "lan_log_p_lan" in data.non_tensor_batch:
            non_tensor_select_keys.append("lan_log_p_lan")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_walk_scores(self, data: DataProto) -> dict[str, torch.Tensor]:
        """
        Compute walk-based per-token importance scores for advantage reweighting.

        Registers output hooks on q_proj and k_proj of every attention layer,
        runs a no-grad forward pass (reusing the same packed / remove-padding
        path as compute_log_prob), then computes the layer-averaged
        Sketch-Determined Walk importance and expands it to response-token level.

        Args:
            data: DataProto with keys ``input_ids``, ``attention_mask``,
                  ``position_ids``, ``responses``, ``response_mask``.

        Returns:
            dict with key ``walk_importance``: tensor (batch_size, response_length).
        """
        walk_cfg = self.config.get("walk_importance", {})
        block_size = walk_cfg.get("block_size", 32)
        hadamard_dim = walk_cfg.get("hadamard_dim", 64)
        walk_degree = walk_cfg.get("walk_degree", 4)
        norm_mode = walk_cfg.get("norm_mode", "relu_max")
        agg_mode = walk_cfg.get("agg_mode", "all")

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "response_mask"]
        data = data.select(batch_keys=[k for k in select_keys if k in data.batch])

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        computer = WalkImportanceComputer(
            block_size=block_size,
            hadamard_dim=hadamard_dim,
            walk_degree=walk_degree,
            norm_mode=norm_mode,
            agg_mode=agg_mode,
        )

        walk_scores_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            input_ids = micro_batch.batch["input_ids"]
            attention_mask = micro_batch.batch["attention_mask"]
            position_ids = micro_batch.batch["position_ids"]
            response_mask = micro_batch.batch["response_mask"]
            batch_size, seqlen = input_ids.shape

            # Compute cu_seqlens from attention_mask for unpacking packed tensors
            seq_lens = attention_mask.sum(dim=1).int()
            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
            cu_seqlens[1:] = seq_lens.cumsum(0)
            max_seqlen = seq_lens.max().item()
            # prompt_lens per sequence: total_len - response_len
            prompt_lens = seq_lens.long() - response_mask.sum(dim=1).long()

            # Register hooks before forward pass
            handles, layer_qk = computer.register_hooks(self.actor_module)

            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                    if self.use_remove_padding:
                        input_ids_rmpad, indices, cu_seqlens_rmpad, *_ = unpad_input(
                            input_ids.unsqueeze(-1), attention_mask
                        )
                        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                        if position_ids.dim() == 3:
                            position_ids_rmpad = (
                                index_first_axis(
                                    rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                                ).transpose(0, 1).unsqueeze(1)
                            )
                        else:
                            position_ids_rmpad = index_first_axis(
                                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                            ).transpose(0, 1)

                        self.actor_module(
                            input_ids=input_ids_rmpad,
                            attention_mask=None,
                            position_ids=position_ids_rmpad,
                            use_cache=False,
                        )
                        # Use the cu_seqlens we computed from attention_mask
                        # (cu_seqlens_rmpad from unpad_input is equivalent)
                    else:
                        self.actor_module(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )

            # Remove hooks immediately after forward pass
            for h in handles:
                h.remove()

            # Compute walk importance scores: (batch_size, response_length)
            scores = computer.compute(
                layer_qk=layer_qk,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                response_mask=response_mask,
                prompt_lens=prompt_lens,
            )
            walk_scores_lst.append(scores)

        walk_importance = torch.cat(walk_scores_lst, dim=0)

        if use_dynamic_bsz:
            walk_importance = restore_dynamic_batch(walk_importance, batch_idx_list)

        return {"walk_importance": walk_importance}

    # ------------------------------------------------------------------
    # PIVOT: temporal walk change scores for loss gating
    # ------------------------------------------------------------------

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_pivot_scores(self, data: DataProto) -> dict[str, torch.Tensor]:
        """
        Compute PIVOT temporal walk change scores (||ΔR_t||) for loss gating.

        Runs the same Q/K hook forward pass as compute_walk_scores, but uses
        TemporalWalkComputer: accumulates walk across TOKEN GENERATION STEPS
        (horizontal/temporal) rather than across transformer layers (vertical).

        At each response block t: score_t = ||a_t @ C_{t-1}||_2 where C is the
        accumulated temporal walk state.  High score → structural change point.

        Returns:
            dict with key ``pivot_scores``: tensor (batch_size, response_length).
        """
        pivot_cfg = self.config.get("pivot", {})
        block_size = pivot_cfg.get("block_size", 32)
        norm_mode = pivot_cfg.get("norm_mode", "relu_max")

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "response_mask"]
        data = data.select(batch_keys=[k for k in select_keys if k in data.batch])

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        computer = TemporalWalkComputer(block_size=block_size, norm_mode=norm_mode)
        pivot_scores_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            input_ids = micro_batch.batch["input_ids"]
            attention_mask = micro_batch.batch["attention_mask"]
            position_ids = micro_batch.batch["position_ids"]
            response_mask = micro_batch.batch["response_mask"]
            batch_size, _ = input_ids.shape

            seq_lens = attention_mask.sum(dim=1).int()
            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
            cu_seqlens[1:] = seq_lens.cumsum(0)
            max_seqlen = seq_lens.max().item()
            prompt_lens = seq_lens.long() - response_mask.sum(dim=1).long()

            handles, layer_qk = computer.register_hooks(self.actor_module)

            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                    if self.use_remove_padding:
                        input_ids_rmpad, indices, *_ = unpad_input(
                            input_ids.unsqueeze(-1), attention_mask
                        )
                        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                        if position_ids.dim() == 3:
                            position_ids_rmpad = (
                                index_first_axis(
                                    rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                                ).transpose(0, 1).unsqueeze(1)
                            )
                        else:
                            position_ids_rmpad = index_first_axis(
                                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                            ).transpose(0, 1)
                        self.actor_module(
                            input_ids=input_ids_rmpad,
                            attention_mask=None,
                            position_ids=position_ids_rmpad,
                            use_cache=False,
                        )
                    else:
                        self.actor_module(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )

            for h in handles:
                h.remove()

            scores = computer.compute(
                layer_qk=layer_qk,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                response_mask=response_mask,
                prompt_lens=prompt_lens,
            )
            pivot_scores_lst.append(scores)

        pivot_scores = torch.cat(pivot_scores_lst, dim=0)
        if use_dynamic_bsz:
            pivot_scores = restore_dynamic_batch(pivot_scores, batch_idx_list)

        return {"pivot_scores": pivot_scores}

    def compute_fork_scores_v2(self, data: DataProto) -> dict[str, torch.Tensor]:
        """
        Compute PIVOT-v2 Phase 1 fork scores via reward-conditioned representation
        divergence.

        Hooks last-layer hidden states during an inference forward pass
        (one hook on the final decoder layer, no Q/K hooks across all layers).
        Groups sequences by prompt and computes:
            v[t] = mean(h_t | correct rollouts) - mean(h_t | wrong rollouts)
            fork_score[t] = ||v[t]||_2  (normalised to [0,1] per response)

        Returns:
            dict with key ``pivot_scores``: tensor (batch_size, response_length).
        """
        pivot_cfg = self.config.get("pivot", {})
        norm_mode = pivot_cfg.get("norm_mode", "relu_max")
        n_per_group = data.meta_info.get("n_rollouts_per_prompt", 8)

        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        # Always use fixed micro-batch to preserve group boundaries for grouping
        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "response_mask", "token_level_scores",
        ]
        data = data.select(batch_keys=[k for k in select_keys if k in data.batch])
        micro_batches = data.split(micro_batch_size)

        computer = RepresentationDivergenceComputer(norm_mode=norm_mode)

        # Collect response hidden states across micro-batches (CPU to bound GPU memory)
        all_h_resp: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            input_ids = micro_batch.batch["input_ids"]
            attention_mask = micro_batch.batch["attention_mask"]
            position_ids = micro_batch.batch["position_ids"]
            response_mask = micro_batch.batch["response_mask"]
            # Scalar reward per sequence: sign of summed token-level scores
            mb_rewards = micro_batch.batch["token_level_scores"].sum(dim=-1).sign()

            batch_size_mb = input_ids.shape[0]
            seq_lens = attention_mask.sum(dim=1).int()
            cu_seqlens = torch.zeros(batch_size_mb + 1, dtype=torch.int32, device=input_ids.device)
            cu_seqlens[1:] = seq_lens.cumsum(0)
            max_seqlen = seq_lens.max().item()
            response_len = response_mask.size(1)

            handle, hidden_store = computer.register_last_layer_hook(self.actor_module)

            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                    if self.use_remove_padding:
                        input_ids_rmpad, indices, *_ = unpad_input(
                            input_ids.unsqueeze(-1), attention_mask
                        )
                        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                        if position_ids.dim() == 3:
                            position_ids_rmpad = (
                                index_first_axis(
                                    rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                                ).transpose(0, 1).unsqueeze(1)
                            )
                        else:
                            position_ids_rmpad = index_first_axis(
                                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                            ).transpose(0, 1)
                        self.actor_module(
                            input_ids=input_ids_rmpad,
                            attention_mask=None,
                            position_ids=position_ids_rmpad,
                            use_cache=False,
                        )
                    else:
                        self.actor_module(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )

            if handle is not None:
                handle.remove()

            h = hidden_store.get("h")
            if h is None:
                hidden_dim_fallback = 1
                all_h_resp.append(torch.zeros(batch_size_mb, response_len, hidden_dim_fallback))
                all_masks.append(response_mask.cpu())
                all_rewards.append(mb_rewards.cpu())
                continue

            # Unpack packed (remove_padding) sequences → (batch, seq_len, hidden_dim)
            if self.use_remove_padding:
                # h may be (1, total_tokens, hidden_dim) when input_ids_rmpad has
                # a leading batch dim of 1; squeeze to (total_tokens, hidden_dim).
                if h.dim() == 3:
                    h = h.squeeze(0)   # (total_tokens, hidden_dim)
                hidden_dim = h.size(-1)
                h_full = torch.zeros(
                    batch_size_mb, max_seqlen, hidden_dim,
                    device=h.device, dtype=h.dtype,
                )
                for i in range(batch_size_mb):
                    s_i = int(cu_seqlens[i].item())
                    e_i = int(cu_seqlens[i + 1].item())
                    h_full[i, :e_i - s_i] = h[s_i:e_i]
            else:
                h_full = h  # (batch_size_mb, max_seqlen, hidden_dim)

            # Extract response portion: last resp_len_i tokens of each sequence
            hidden_dim = h_full.size(-1)
            h_resp = torch.zeros(
                batch_size_mb, response_len, hidden_dim,
                device=h_full.device, dtype=h_full.dtype,
            )
            for i in range(batch_size_mb):
                resp_len_i = int(response_mask[i].sum().item())
                if resp_len_i == 0:
                    continue
                seq_len_i = int(seq_lens[i].item())
                tok_start = max(0, seq_len_i - resp_len_i)
                actual = seq_len_i - tok_start
                h_resp[i, :actual] = h_full[i, tok_start:seq_len_i]

            all_h_resp.append(h_resp.cpu())
            all_masks.append(response_mask.cpu())
            all_rewards.append(mb_rewards.cpu())

        # Stack full (local) batch and compute group-level fork scores
        h_resp_full = torch.cat(all_h_resp, dim=0)    # (B_local, response_len, hidden_dim)
        masks_full = torch.cat(all_masks, dim=0)        # (B_local, response_len)
        rewards_full = torch.cat(all_rewards, dim=0)   # (B_local,)

        pivot_scores = computer.compute_from_hidden(
            h_resp=h_resp_full,
            response_mask=masks_full,
            rewards=rewards_full,
            n_per_group=n_per_group,
        ).to(get_device_id())

        return {"pivot_scores": pivot_scores}

    # ------------------------------------------------------------------
    # PIVOT: Langevin generation using actor model (walk-triggered)
    # ------------------------------------------------------------------

    @staticmethod
    def _langevin_step(logits: torch.Tensor, eta: float, sigma: float) -> torch.Tensor:
        """One Langevin step: gradient of −H(softmax(logits)) + Gaussian noise.
        Sharpens the distribution while preventing mode collapse.
        logits: (B, vocab_size) or (vocab_size,)
        """
        p = torch.softmax(logits.float(), dim=-1)
        log_p = torch.log_softmax(logits.float(), dim=-1)
        H = -(p * log_p).sum(dim=-1, keepdim=True)   # (B, 1) or scalar
        grad_neg_H = p * (log_p + H)                  # ∂(−H)/∂logits
        logits = logits.float() + eta * grad_neg_H + sigma * torch.randn_like(logits.float())
        return logits.to(torch.bfloat16)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def generate_with_langevin(self, data: DataProto) -> dict[str, torch.Tensor]:
        """
        Token-by-token generation with walk-triggered Langevin sampling (PIVOT).

        At each generation step:
          1. Forward pass with KV cache → logits + Q/K from hooks.
          2. Compute block-level attention a_t from Q (hook) and K (KV cache).
          3. Temporal walk update: multi = a_t @ C; C[b_t] = a_t + multi.
          4. trigger = ||multi||.mean(dim=-1) — structural change signal.
          5. If trigger > threshold: run K Langevin steps on logits.
          6. Sample next token; record position in P_causal if Langevin ran.

        Returns dict with:
            ``responses``    : (B, max_new_tokens) generated token ids
            ``p_causal_mask``: (B, max_new_tokens) 1 where Langevin ran, else 0
        """
        pivot_cfg = self.config.get("pivot", {})
        block_size = pivot_cfg.get("block_size", 32)
        threshold = pivot_cfg.get("langevin_threshold", 0.1)
        K = pivot_cfg.get("langevin_K", 3)
        eta = pivot_cfg.get("langevin_eta", 0.05)
        sigma = pivot_cfg.get("langevin_sigma", 0.02)
        max_new_tokens = pivot_cfg.get("max_new_tokens", 512)

        self.actor_module.eval()

        input_ids = data.batch["input_ids"].to(get_device_id())           # (B, prompt_len)
        attention_mask = data.batch["attention_mask"].to(get_device_id()) # (B, prompt_len)
        B, prompt_len = input_ids.shape

        max_seq_len = prompt_len + max_new_tokens
        max_num_blocks = (max_seq_len + block_size - 1) // block_size

        # Temporal walk state: C[b, t_block, j] = walk row for block t_block
        C = torch.zeros(B, max_num_blocks, max_num_blocks,
                        device=input_ids.device, dtype=torch.float32)

        generated = []
        p_causal = []
        past_kv = None

        computer = TemporalWalkComputer(block_size=block_size)
        # Register model config once so _unpack_projection has head dims
        _, _ = computer.register_hooks(self.actor_module)  # dummy to set model config
        # (handles not needed here; we use a different path below)

        with torch.no_grad():
            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                for step in range(max_new_tokens):
                    cur_input = input_ids if step == 0 else next_token

                    # --- forward pass with Q/K hooks ---
                    layer_qk: dict[int, dict[str, torch.Tensor]] = {}
                    handles: list = []

                    model_inner = self.actor_module
                    while hasattr(model_inner, "module"):
                        model_inner = model_inner.module

                    layer_counter = [0]
                    for name, mod in model_inner.named_modules():
                        if not (hasattr(mod, "q_proj") and hasattr(mod, "k_proj")):
                            continue
                        if any(hasattr(c, "q_proj") for c in mod.children()):
                            continue
                        idx = layer_counter[0]
                        layer_counter[0] += 1

                        def _q_hook(m, inp, out, i=idx):
                            layer_qk.setdefault(i, {})["q"] = out.detach()

                        def _k_hook(m, inp, out, i=idx):
                            layer_qk.setdefault(i, {})["k"] = out.detach()

                        handles.append(mod.q_proj.register_forward_hook(_q_hook))
                        handles.append(mod.k_proj.register_forward_hook(_k_hook))

                    outputs = self.actor_module(
                        input_ids=cur_input,
                        attention_mask=attention_mask if step == 0 else None,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    for h in handles:
                        h.remove()

                    logits = outputs.logits[:, -1, :]  # (B, vocab)
                    past_kv = outputs.past_key_values

                    # --- block-level attention for current token ---
                    # Q from hook: (B, Hq*D) — just the new token's projection
                    # K_full from KV cache: (B, Hkv, T, D)
                    cur_total_len = prompt_len + step + 1
                    t_block = (prompt_len + step) // block_size

                    if layer_qk:
                        layer_attn_rows = []
                        Hq = computer.num_q_heads
                        Hkv = computer.num_kv_heads
                        D = computer.head_dim

                        for li in sorted(layer_qk.keys()):
                            q_packed = layer_qk[li]["q"]  # (B, Hq*D)
                            # Reshape Q: (B, Hq, D) → mean over heads → (B, D)
                            q_vec = q_packed.view(B, Hq, D).mean(dim=1)  # (B, D)

                            # K_full from KV cache
                            try:
                                if hasattr(past_kv, "key_cache"):
                                    k_full = past_kv.key_cache[li]  # (B, Hkv, T, D)
                                elif isinstance(past_kv, tuple):
                                    k_full = past_kv[li][0]
                                else:
                                    k_full = past_kv[li][0]
                                # Mean over KV heads: (B, T, D)
                                k_mean = k_full.mean(dim=1)
                                # Block-pool T positions: (B, num_cur_blocks, D)
                                T = k_mean.size(1)
                                num_cur_blocks = (T + block_size - 1) // block_size
                                pad = num_cur_blocks * block_size - T
                                if pad:
                                    k_mean = F.pad(k_mean, (0, 0, 0, pad))
                                k_blocked = k_mean.view(B, num_cur_blocks, block_size, D).mean(dim=2)
                                # Attention: q_vec (B, D) · k_blocked (B, num_cur_blocks, D)
                                scale = D ** -0.5
                                attn_logits = torch.einsum("bd,bnd->bn", q_vec, k_blocked) * scale
                                attn_row = torch.softmax(attn_logits, dim=-1)  # (B, num_cur_blocks)
                                layer_attn_rows.append(attn_row)
                            except Exception:
                                pass

                        if layer_attn_rows:
                            a_t_short = torch.stack(layer_attn_rows).mean(0)  # (B, num_cur_blocks)
                            # Embed into full-size vector
                            a_t = torch.zeros(B, max_num_blocks, device=C.device, dtype=torch.float32)
                            n = min(a_t_short.size(1), max_num_blocks)
                            a_t[:, :n] = a_t_short[:, :n].float()

                            # Temporal walk update
                            multi = torch.bmm(a_t.unsqueeze(1), C).squeeze(1)  # (B, max_num_blocks)
                            tb = min(t_block, max_num_blocks - 1)
                            C[:, tb, :] = (a_t + multi)
                            trigger = multi.norm(dim=-1)  # (B,)
                        else:
                            trigger = torch.zeros(B, device=logits.device)
                    else:
                        trigger = torch.zeros(B, device=logits.device)

                    # --- Langevin at high-trigger positions ---
                    is_causal = (trigger > threshold)  # (B,)
                    p_causal.append(is_causal.long())

                    if is_causal.any():
                        mod_logits = logits.clone()
                        for _ in range(K):
                            mod_logits = self._langevin_step(mod_logits, eta=eta, sigma=sigma)
                        # Apply Langevin only to triggered sequences
                        logits = torch.where(is_causal.unsqueeze(1), mod_logits, logits)

                    # --- sample next token ---
                    probs = torch.softmax(logits.float(), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                    generated.append(next_token)

                    # Update attention_mask for cache step
                    if step == 0:
                        attention_mask = torch.ones(
                            B, prompt_len + 1, device=input_ids.device,
                            dtype=attention_mask.dtype,
                        )
                    else:
                        attention_mask = torch.cat(
                            [attention_mask,
                             torch.ones(B, 1, device=input_ids.device, dtype=attention_mask.dtype)],
                            dim=1,
                        )

        responses = torch.cat(generated, dim=1)          # (B, max_new_tokens)
        p_causal_mask = torch.stack(p_causal, dim=1)     # (B, max_new_tokens)

        return {"responses": responses, "p_causal_mask": p_causal_mask}

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        if "lan_trigger_mask" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("lan_trigger_mask")
        if "lan_log_p_lan" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("lan_log_p_lan")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        # Release any fragmented cached memory from the preceding PIVOT forward pass
        # or log-prob computation before the gradient-bearing update loop.
        torch.cuda.empty_cache()

        self._pivot_update_step += 1
        _pivot_cfg_train = getattr(self.config, "pivot", None) or {}
        _lan_grpo_coeff_train = float(_pivot_cfg_train.get("lan_grpo_coeff", 0.0))
        # Restrict pg_loss to Langevin-triggered positions only (exact alignment).
        # When True: response_mask = response_mask & lan_grpo_mask, so gradient
        # flows only where Langevin fired.
        _lan_grpo_restrict_to_trigger = bool(_pivot_cfg_train.get("lan_grpo_restrict_to_trigger", False))
        # Soft IS off-policy correction (DRIFT): the PPO ratio stays π_θ/π_old (trust
        # region intact); the correction is a per-token advantage reweight
        # w = min(1, π_old/π_lan_old) at trigger positions (pg = -A·ratio, so
        # reweighting A is equivalent to reweighting pg).
        _lan_grpo_soft_is = bool(_pivot_cfg_train.get("lan_grpo_soft_is", False))

        # v21 IS-weight accumulators (rank-local, emitted once at the end of
        # update_policy so each rank contributes a single scalar per metric and
        # the gathered list across DP ranks is flat — avoids the ragged list-of-
        # lists that np.mean can't handle when micro-batch counts differ per rank).
        _v21_w_sum: float = 0.0
        _v21_w_sq_sum: float = 0.0
        _v21_w_min: float = float("inf")
        _v21_log_w_sum: float = 0.0
        _v21_n_trig_total: int = 0
        _v21_samples: list[float] = []  # for percentile estimation

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    full_response_mask = response_mask  # preserved for metrics unaffected by entropy_top_ratio masking
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    entropy_top_ratio = self.config.get("entropy_top_ratio", None)

                    calculate_entropy = (
                        self.config.calculate_entropy or (entropy_coeff != 0) or (entropy_top_ratio is not None)
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # High-Entropy RL: restrict policy-gradient updates to the top-ρ
                    # response tokens by per-token entropy from the current policy.
                    # Adapted from Wang et al., "Beyond the 80/20 Rule", NeurIPS 2025.
                    # https://arxiv.org/abs/2506.01939
                    if entropy_top_ratio is not None and entropy is not None:
                        entropy_top_mask = get_global_entropy_top_mask(
                            entropy=entropy,
                            response_mask=response_mask,
                            top_ratio=entropy_top_ratio,
                        )
                        response_mask = response_mask * entropy_top_mask
                        micro_batch_metrics["actor/high_ent_token_frac"] = (
                            response_mask.sum() / model_inputs["response_mask"].float().sum().clamp(min=1)
                        ).item()

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    if "lan_grpo_mask" in outputs and _lan_grpo_coeff_train > 0.0 and _lan_grpo_soft_is:
                        # Soft IS multiplier. PPO ratio stays π_θ/π_old; the off-policy
                        # correction is a per-token weight w = min(1, π_old/π_lan_old) at
                        # trigger positions, applied by reweighting advantages
                        # (pg = -A·ratio → multiplying A is equivalent to multiplying pg).
                        # No log_prob / old_log_prob patch — trust region stays intact and
                        # we fall through to standard PPO with the reweighted advantages.
                        _lg_mask = outputs["lan_grpo_mask"].float()
                        _log_p_lan = outputs["lan_grpo_log_p"]
                        _log_pi_old = old_log_prob.detach()
                        _log_w_v21 = (_log_pi_old - _log_p_lan).clamp(max=0.0)
                        _w_corr_v21 = torch.exp(_log_w_v21)
                        _is_weight_v21 = (1.0 - _lg_mask) + _w_corr_v21 * _lg_mask
                        advantages = advantages * _is_weight_v21
                        # Accumulate IS-weight stats into rank-local scalars; emit once at
                        # the end of update_policy so the gathered per-rank metric is a flat
                        # 8-element list (not ragged 8×N_micro_batches that np.mean can't
                        # reduce when N differs per rank — e.g. due to overlong-prompt drops).
                        _trig_bool = _lg_mask.bool()
                        if _trig_bool.any():
                            _w_trig = _w_corr_v21[_trig_bool].float()
                            _log_w_trig = _log_w_v21[_trig_bool].float()
                            _v21_w_sum += _w_trig.sum().item()
                            _v21_log_w_sum += _log_w_trig.sum().item()
                            _v21_n_trig_total += int(_w_trig.numel())
                            _v21_w_min = min(_v21_w_min, float(_w_trig.min().item()))
                            # Reservoir-ish: cap at 4096 samples to bound memory.
                            if len(_v21_samples) < 4096:
                                _v21_samples.extend(_w_trig.detach().cpu().tolist())

                    if _lan_grpo_restrict_to_trigger and "lan_grpo_mask" in outputs:
                        response_mask = response_mask * outputs["lan_grpo_mask"]

                    # v14 fork-weighted advantage: amplify gradient at trigger positions.
                    # Fork positions (high delta_var) are where trajectory outcomes are causally
                    # decided — advantage signal there carries more credit-assignment information.
                    _fork_alpha = float(_pivot_cfg_train.get("fork_advantage_alpha", 0.0))
                    if _fork_alpha > 0.0 and "lan_grpo_mask" in outputs:
                        _fork_mask = outputs["lan_grpo_mask"].float()
                        advantages = advantages * (1.0 + _fork_alpha * _fork_mask)

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss

                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=full_response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            # Gate entropy bonus on positions where rollouts diverged AND
                            # the group has learning signal (non-zero GRPO advantages).
                            # Total entropy loss contribution is capped at entropy_loss_cap
                            # to prevent runaway: even at high entropy/trigger_frac the
                            # entropy term cannot overwhelm the policy gradient.
                            n = data.meta_info.get("n_rollouts", 1)
                            responses = model_inputs["responses"]  # (B, response_len)
                            if n > 1 and responses.shape[0] % n == 0:
                                B, L = responses.shape
                                G = B // n
                                grouped = responses.view(G, n, L)

                                # binary disagreement: 1 where at least two rollouts differ
                                all_same = (grouped == grouped[:, :1, :]).all(dim=1)  # (G, L)
                                disagree = (~all_same).float()  # (G, L)

                                # group-level learning signal: any rollout has non-zero advantage
                                grouped_adv = advantages.view(G, n, L)  # (G, n, L)
                                has_signal = (grouped_adv.abs().amax(dim=(1, 2)) > 1e-4).float()  # (G,)

                                group_gate = disagree * has_signal.unsqueeze(1)  # (G, L)
                                gate = group_gate.unsqueeze(1).expand(-1, n, -1).reshape(B, L) * response_mask
                            else:
                                gate = torch.zeros_like(response_mask)
                            micro_batch_metrics["pivot/trigger_gate_mode"] = 1.0
                            micro_batch_metrics["pivot/trigger_frac"] = (
                                gate.sum() / response_mask.sum().clamp(min=1)
                            ).item()
                            micro_batch_metrics["pivot/entropy_loss"] = 0.0
                            if gate.sum() > 0:
                                entropy_at_trigger = agg_loss(
                                    loss_mat=entropy, loss_mask=gate, loss_agg_mode=loss_agg_mode
                                )
                                entropy_loss = entropy_at_trigger * entropy_coeff
                                # Cap total entropy contribution so it cannot overwhelm pg_loss.
                                entropy_loss_cap = self.config.entropy_loss_cap
                                if entropy_loss_cap > 0:
                                    entropy_loss = entropy_loss.clamp(max=entropy_loss_cap)
                                micro_batch_metrics["pivot/entropy_loss"] = entropy_loss.detach().item()
                                policy_loss -= entropy_loss

                    if _lan_grpo_coeff_train > 0.0:
                        _n_trig = outputs["lan_grpo_mask"].sum().item() if "lan_grpo_mask" in outputs else 0
                        _n_resp = full_response_mask.sum().item()  # use unmasked denominator so entropy_top_ratio doesn't inflate trig_frac
                        micro_batch_metrics["pivot/lan_grpo_trig_frac"] = _n_trig / max(_n_resp, 1)
                        for _k, _v in outputs.get("dv_stats", {}).items():
                            micro_batch_metrics[f"pivot/dv_{_k}"] = _v

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        # v21: emit IS-weight summary once per update_policy call (per rank).
        # Each rank contributes a single scalar per metric → gathered list across
        # DP ranks is flat → np.mean reduces cleanly regardless of per-rank
        # micro-batch count variation.
        if _lan_grpo_soft_is:
            if _v21_n_trig_total > 0:
                _v21_w_mean = _v21_w_sum / _v21_n_trig_total
                _v21_log_w_mean = _v21_log_w_sum / _v21_n_trig_total
                _v21_samples.sort()
                _v21_p10 = _v21_samples[max(0, int(len(_v21_samples) * 0.1) - 1)]
            else:
                _v21_w_mean = 1.0
                _v21_log_w_mean = 0.0
                _v21_w_min = 1.0
                _v21_p10 = 1.0
            v21_metrics = {
                "pivot/v21_w_corr_mean": _v21_w_mean,
                "pivot/v21_w_corr_min": _v21_w_min,
                "pivot/v21_w_corr_p10": _v21_p10,
                "pivot/v21_log_w_mean": _v21_log_w_mean,
                "pivot/v21_trig_count": _v21_n_trig_total,
            }
            append_to_dict(metrics, v21_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
