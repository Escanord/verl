# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
PIVOT vLLM Patch: Temporal Walk Computation During Autoregressive Rollout
=========================================================================

Implements Phase 2 of PIVOT (Pivot-point Identification Via On-the-fly walk
Tracking): Langevin-augmented sampling triggered by temporal walk change signal
||ΔR_t||, computed entirely within the vLLM decode loop.

Architecture
------------
During autoregressive decode, each forward pass processes one new token per
sequence in the batch. We register forward pre-hooks on every attention layer
to capture Q_new and K_new (the new token's query and key) for each layer.
After all layers fire, we average Q/K across layers and store the result in
`_pivot_decode_state`.

One `PIVOTRolloutProcessor` instance (a vLLM logits_processor) is created per
generation request. It:
  1. Reads its sequence's Q_new, K_new from `_pivot_decode_state` using a
     shared batch cursor (incremented in logits_processor call order).
  2. Appends K_new to its own K history (shadow KV buffer for response tokens).
  3. Computes block-level attention from Q_new to all past blocks in K_history.
  4. Updates the temporal walk matrix C and computes ||ΔR_t||.
  5. If ||ΔR_t|| > threshold, applies Langevin entropy-maximising steps to logits.

Signal scope: only tracks causal structure WITHIN the generated response
(K_history grows from the first response token). This is intentional — we're
measuring how the model's self-attention structure evolves as it generates.

Assumption: vLLM calls logits_processors in the same order as sequences in the
decode batch. This is consistent with vLLM v1's sequential sampling loop.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level shared decode state (written by attention hooks, read by
# PIVOTRolloutProcessor instances).
# ---------------------------------------------------------------------------


@dataclass
class _PIVOTDecodeState:
    """Shared state updated once per forward pass, read by logits processors."""
    q_batch: Optional[torch.Tensor] = None   # (batch, q_dim) — layer-averaged
    k_batch: Optional[torch.Tensor] = None   # (batch, k_dim) — layer-averaged
    cursor: int = 0                          # next batch position to serve
    num_layers: int = 0                      # total layers being tracked
    # Accumulators for layer averaging (cleared after all layers fire)
    _layer_q: List[torch.Tensor] = field(default_factory=list)
    _layer_k: List[torch.Tensor] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_pivot_decode_state = _PIVOTDecodeState()

# Registry: maps tuple(response_token_ids) -> per-token trigger values list[float].
# A value > 0 at position t means Langevin fired at that response token.
# Populated by PIVOTv2LangevinAdapter.update_state when requests complete;
# consumed and cleared by vllm_async_server.generate() after final_res arrives.
_TRIGGER_REGISTRY: dict = {}


def _reset_pivot_state():
    """Reset the module-level decode state (call before each generation batch)."""
    s = _pivot_decode_state
    with s.lock:
        s.q_batch = None
        s.k_batch = None
        s.cursor = 0
        s._layer_q.clear()
        s._layer_k.clear()


# ---------------------------------------------------------------------------
# Langevin helper (shared with dp_actor.py logic)
# ---------------------------------------------------------------------------


def _identify_eos_class_tokens(tokenizer) -> List[int]:
    """Tokens whose generation terminates a response.

    Union of (i) tokenizer.eos_token_id(s) and (ii) common chat-end markers that
    happen to tokenize to a single token id.  Used to mask EOS-class tokens out
    of the Langevin top-K subspace so perturbations cannot push toward early
    response termination (the §3.4 length-collapse pathway).
    """
    out: set = set()
    if tokenizer is None:
        return []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        try:
            out.add(int(eos))
        except (TypeError, ValueError):
            pass
    # Some tokenizers expose multiple eos ids; the attribute may be either
    # a list/tuple/iterable, a single int, or absent entirely.
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is not None:
        if isinstance(eos_ids, int):
            out.add(int(eos_ids))
        else:
            try:
                for t in eos_ids:
                    try:
                        out.add(int(t))
                    except (TypeError, ValueError):
                        pass
            except TypeError:
                pass  # not iterable, not int — give up silently
    # Chat-template end markers — only include when they map to a single token.
    for marker in ("<|im_end|>", "<|endoftext|>", "<|end|>", "<|eos|>"):
        try:
            ids = tokenizer.encode(marker, add_special_tokens=False)
            if isinstance(ids, list) and len(ids) == 1:
                out.add(int(ids[0]))
        except Exception:
            pass
    return sorted(out)


def _apply_eos_mask_(logits: torch.Tensor, eos_ids_tensor: Optional[torch.Tensor]) -> torch.Tensor:
    """Set EOS-class token logits to -inf in-place.  No-op if eos_ids_tensor is None.

    Accepts any logits shape ending in the vocab axis; eos_ids_tensor is a 1-D
    long tensor of token ids to mask.
    """
    if eos_ids_tensor is None or eos_ids_tensor.numel() == 0:
        return logits
    eos_dev = eos_ids_tensor.to(logits.device)
    logits.index_fill_(-1, eos_dev, float("-inf"))
    return logits


def _langevin_step(logits: torch.Tensor, eta: float, sigma: float) -> torch.Tensor:
    """One step of entropy-gradient Langevin: logits += eta*∇H + N(0,σ²).
    Preserves -inf masks so top-k constraints are respected across steps.
    """
    mask = logits.float() == float('-inf')
    p = torch.softmax(logits.float(), dim=-1)
    log_p = torch.log(p.clamp(min=1e-12))  # 0*log(0)=0 convention
    H = -(p * log_p).sum(dim=-1, keepdim=True)
    grad_H = -p * (log_p + H)  # ∂H/∂logit_i = p_i(-log_p_i - H)
    noisy = logits.float() + eta * grad_H + sigma * torch.randn_like(logits.float())
    noisy = torch.where(mask, torch.full_like(noisy, float('-inf')), noisy)
    return noisy.to(logits.dtype)


def _langevin_sample(
    logits: torch.Tensor,
    K: int,
    eta: float,
    sigma: float,
    top_k: int = 0,
    eos_ids_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply K Langevin steps, optionally constrained to top-k tokens.

    top_k=0 means no constraint (full vocabulary).
    top_k>0 masks all but the top-k tokens to -inf before stepping,
    preventing Langevin from activating incoherent low-rank tokens.

    eos_ids_tensor (optional): token ids masked to -inf BEFORE top-k selection,
    so the perturbation subspace can never include EOS-class tokens.
    """
    if eos_ids_tensor is not None and eos_ids_tensor.numel() > 0:
        logits = logits.clone()
        _apply_eos_mask_(logits, eos_ids_tensor)
    if top_k > 0:
        safe_logits = logits.float().nan_to_num(nan=float('-inf'))
        topk_vals, topk_idx = torch.topk(safe_logits, min(top_k, logits.size(-1)))
        masked = torch.full_like(safe_logits, float('-inf'))
        masked.scatter_(-1, topk_idx, topk_vals)
        logits = masked.to(logits.dtype)
    for _ in range(K):
        logits = _langevin_step(logits, eta, sigma)
    return logits


def _mala_step(logits: torch.Tensor, eta: float, sigma: float) -> torch.Tensor:
    """One MALA step: Langevin proposal + entropy-MH accept/reject.

    Operates only on finite (non -inf) logit dimensions so top-k masks are
    respected.  Uses a pure entropy Metropolis criterion (log α = H_prop - H_cur)
    rather than the full MALA q-ratio correction.  The q-ratio correction requires
    σ ~ η to be numerically well-conditioned; when σ << η (e.g. σ=0.01, η=0.3)
    the 1/(2σ²) amplifier makes it blow up to ~O(100), rendering the criterion
    vacuous (acceptance rate ≈ 100%).  The entropy-MH rule gives meaningful
    rejection of entropy-decreasing proposals without touching σ or η.
    """
    inf_mask = ~torch.isfinite(logits.float())
    logits_f = logits.float()

    # Current state
    p = torch.softmax(logits_f, dim=-1)
    log_p = torch.log(p.clamp(min=1e-12))
    H_cur = -(p * log_p).sum()
    grad_H = -p * (log_p + H_cur)

    # Proposal: only add noise/gradient on active (non -inf) dimensions
    noise = torch.zeros_like(logits_f)
    active = ~inf_mask
    n_active = int(active.sum().item())
    if n_active > 0:
        noise[active] = sigma * torch.randn(n_active, device=logits.device, dtype=logits_f.dtype)
    logits_prop = (logits_f + eta * grad_H + noise).masked_fill(inf_mask, float('-inf'))

    # Proposed entropy
    p_prop = torch.softmax(logits_prop, dim=-1)
    log_p_prop = torch.log(p_prop.clamp(min=1e-12))
    H_prop = -(p_prop * log_p_prop).sum()

    # Entropy-MH acceptance: always accept entropy-increasing moves; reject
    # entropy-decreasing moves with probability 1 - exp(H_prop - H_cur).
    log_alpha = float(H_prop - H_cur)
    accepted = torch.log(torch.rand(1, device=logits.device)).item() < log_alpha

    if accepted:
        return logits_prop.to(logits.dtype), True
    return logits, False


def _langevin_step_feedback(
    logits: torch.Tensor,
    sigma: float,
    top_k: int,
    G: Optional[torch.Tensor],
    alpha: float,
    eos_ids_tensor: Optional[torch.Tensor] = None,
) -> tuple:
    """Langevin step with G-guided adaptive noise. No entropy-gradient drift.

    The noise direction is a convex mix of the current G direction (exploitation)
    and a fresh random vector (exploration):

        ε = α·(G/‖G‖) + (1-α)·randn_unit    (projected onto active top-k subspace)
        logits_active += σ·ε

    Returns (new_logits, eps_full_vocab) where eps_full_vocab is saved by the
    caller so that G can be updated at the next step using the observed entropy.

    eos_ids_tensor (optional): EOS-class token ids masked to -inf BEFORE top-K
    selection, so the perturbation subspace excludes response-terminating tokens.
    """
    if eos_ids_tensor is not None and eos_ids_tensor.numel() > 0:
        logits = logits.clone()
        _apply_eos_mask_(logits, eos_ids_tensor)
    if top_k > 0:
        safe_logits = logits.float().nan_to_num(nan=float('-inf'))
        topk_vals, topk_idx = torch.topk(safe_logits, min(top_k, logits.size(-1)))
        masked = torch.full_like(safe_logits, float('-inf'))
        masked.scatter_(-1, topk_idx, topk_vals)
        logits = masked.to(logits.dtype)

    inf_mask = ~torch.isfinite(logits.float())
    active_mask = ~inf_mask

    # Full-vocab masked noise — eliminates .item() for n_active (sync 1) and all
    # Python norm-guard branches (syncs 2-4). active_mask.float().sum().sqrt()
    # stays on GPU as a 0-dim tensor; Python never sees the scalar.
    active_f = active_mask.float()
    random_eps = torch.randn_like(logits.float()).mul_(active_f)
    random_eps = random_eps / random_eps.norm().clamp(min=1e-8)

    # Mix G direction with random
    if G is not None:
        # Out-of-place mul so self._G is not corrupted by the active mask.
        G_proj = G.to(logits.device).float() * active_f
        G_unit = G_proj / G_proj.norm().clamp(min=1e-6)
        eps = alpha * G_unit + (1.0 - alpha) * random_eps
        eps = eps / eps.norm().clamp(min=1e-8)
    else:
        eps = random_eps

    # Scale to match standard Gaussian magnitude in active subspace.
    # active_f.sum().sqrt() is a 0-dim GPU tensor — no .item() needed.
    eps = eps * active_f.sum().sqrt()

    logits_f = logits.float()
    logits_f = logits_f + sigma * eps
    logits_f = logits_f.masked_fill(inf_mask, float('-inf'))
    return logits_f.to(logits.dtype), eps


def _langevin_step_feedback_batched(
    logits: torch.Tensor,  # (N, vocab)
    sigma: float,
    top_k: int,
    G: torch.Tensor,       # (N, vocab) — zeros where G is absent
    alpha: float,
    eos_ids_tensor: Optional[torch.Tensor] = None,
) -> tuple:
    """Batched version of _langevin_step_feedback for N triggered sequences.

    Single randn call over (N, vocab) instead of N separate calls — better GPU
    utilization via larger kernels and fewer launch overheads.

    eos_ids_tensor (optional): EOS-class token ids masked to -inf BEFORE top-K
    selection.  See _langevin_step_feedback for rationale.
    """
    if eos_ids_tensor is not None and eos_ids_tensor.numel() > 0:
        logits = logits.clone()
        _apply_eos_mask_(logits, eos_ids_tensor)
    if top_k > 0:
        safe_logits = logits.float().nan_to_num(nan=float('-inf'))
        topk_vals, topk_idx = torch.topk(safe_logits, min(top_k, logits.shape[-1]), dim=-1)
        masked = torch.full_like(safe_logits, float('-inf'))
        masked.scatter_(-1, topk_idx, topk_vals)
        logits = masked.to(logits.dtype)

    inf_mask = ~torch.isfinite(logits.float())   # (N, vocab)
    active_f = (~inf_mask).float()               # (N, vocab)

    # One randn for all N sequences — single large kernel vs N small ones.
    random_eps = torch.randn_like(logits.float()).mul_(active_f)
    row_norm = random_eps.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    random_eps = random_eps / row_norm

    # Per-row G mixing — sequences where G is all-zero stay pure random.
    G_has_signal = G.abs().sum(dim=-1, keepdim=True) > 1e-6  # (N, 1) bool
    G_proj = G.float() * active_f  # out-of-place — _G_stack row must not be mutated
    G_unit = G_proj / G_proj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    eps_G = alpha * G_unit + (1.0 - alpha) * random_eps
    eps = torch.where(G_has_signal, eps_G, random_eps)
    eps = eps / eps.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Scale to active-subspace magnitude — stays on GPU (no .item()).
    n_active = active_f.sum(dim=-1, keepdim=True).sqrt()  # (N, 1)
    eps = eps * n_active

    logits_f = logits.float() + sigma * eps
    logits_f = logits_f.masked_fill(inf_mask, float('-inf'))
    return logits_f.to(logits.dtype), eps


def _pivot_v18_graph_kernel(
    logits: torch.Tensor,          # (N, V) float32 input
    noise: torch.Tensor,           # (N, V) pre-filled randn — consumed each step
    G_stack: torch.Tensor,         # (N, V) current G per slot; zeros where absent
    step_counts: torch.Tensor,     # (N,)  int64
    has_triggered: torch.Tensor,   # (N,)  bool — whether slot has seen a trigger
    H_first: torch.Tensor,         # (N,)  float32 — entropy at first trigger (0 if never)
    sigma: float,
    top_k: int,
    exploit_ratio: float,
    entropy_threshold: "torch.Tensor",  # () float32 GPU scalar — updated between replays
    min_trigger_position: int,
    alpha_target: float,
) -> tuple:
    """CUDA-graph-compatible PIVOT v18 Langevin step (all inputs float32).

    Returns (new_logits, trigger_mask, eps_out, H_batch).
    Non-triggered rows return their original logits unchanged.
    """
    p = torch.softmax(logits, dim=-1)
    H_batch = -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1)

    trigger_mask = (H_batch > entropy_threshold) & (step_counts >= min_trigger_position)

    logits_f = logits
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(logits_f, min(top_k, logits_f.shape[-1]), dim=-1)
        masked = torch.full_like(logits_f, float("-inf"))
        masked.scatter_(-1, topk_idx, topk_vals)
        logits_f = masked

    inf_mask = ~torch.isfinite(logits_f)
    active_f = (~inf_mask).float()

    noise_masked = noise * active_f
    row_norm = noise_masked.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    random_eps = noise_masked / row_norm

    G_has_signal = G_stack.abs().sum(dim=-1, keepdim=True) > 1e-6
    G_proj = G_stack * active_f
    G_unit = G_proj / G_proj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    eps_G = exploit_ratio * G_unit + (1.0 - exploit_ratio) * random_eps
    eps_mixed = torch.where(G_has_signal, eps_G, random_eps)
    eps_mixed = eps_mixed / eps_mixed.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    n_active = active_f.sum(dim=-1, keepdim=True).sqrt()
    eps_scaled = eps_mixed * n_active

    triggered_out = (logits_f + sigma * eps_scaled).masked_fill(inf_mask, float("-inf"))
    logits_new = torch.where(trigger_mask.unsqueeze(-1), triggered_out, logits)
    eps_out = eps_scaled * trigger_mask.float().unsqueeze(-1)

    return logits_new, trigger_mask, eps_out, H_batch


def _pivot_v18_update_G(
    G_state: torch.Tensor,          # (N, V) modified in-place
    H_first: torch.Tensor,          # (N,) modified in-place
    has_triggered: torch.Tensor,    # (N,) bool modified in-place
    prev_eps: torch.Tensor,         # (N, V) eps from previous step
    trigger_mask: torch.Tensor,     # (N,) bool — rows that fired this step
    H_batch: torch.Tensor,          # (N,) entropy from this step (GPU tensor)
    alpha_target: float,
    gamma: float,
) -> None:
    """Update G momentum state in-place.  No Python branches — safe outside CUDA graph."""
    first_time = trigger_mask & ~has_triggered
    H_first.copy_(torch.where(first_time, H_batch, H_first))
    has_triggered.logical_or_(trigger_mask)
    H_target = H_first * alpha_target
    signal = (H_batch - H_target) * trigger_mask.float()
    feedback = signal.unsqueeze(-1) * prev_eps
    G_state.mul_(gamma).add_((1.0 - gamma) * feedback)


def _langevin_step_momentum(
    logits: torch.Tensor,
    eta: float,
    sigma: float,
    gamma: float,
    velocity: torch.Tensor,
    sq_velocity: Optional[torch.Tensor] = None,
    beta2: float = 0.999,
    eps: float = 1e-8,
    top_k: int = 0,
) -> tuple:
    """One Langevin step with momentum velocity that persists across triggers.

    Returns (new_logits, new_velocity) or (new_logits, new_velocity, new_sq_velocity)
    when sq_velocity is provided (Adam-style RMS normalization).

    Momentum formulation:
        step    = η·∇H + σ·noise
        v_new   = γ·v_old + step
        update  = v_new                          (no normalization)
                = v_new / (√s_new + ε)           (with RMS normalization)
        s_new   = β₂·s_old + (1-β₂)·step²       (second moment)
    """
    if top_k > 0:
        safe_logits = logits.float().nan_to_num(nan=float('-inf'))
        topk_vals, topk_idx = torch.topk(safe_logits, min(top_k, logits.size(-1)))
        masked = torch.full_like(safe_logits, float('-inf'))
        masked.scatter_(-1, topk_idx, topk_vals)
        logits = masked.to(logits.dtype)

    inf_mask = ~torch.isfinite(logits.float())
    logits_f = logits.float()

    p = torch.softmax(logits_f, dim=-1)
    log_p = torch.log(p.clamp(min=1e-12))
    H = -(p * log_p).sum()
    grad_H = -p * (log_p + H)

    noise = sigma * torch.randn_like(logits_f)
    noise = noise.masked_fill(inf_mask, 0.0)

    step = eta * grad_H + noise
    new_velocity = gamma * velocity + step
    new_velocity = new_velocity.masked_fill(inf_mask, 0.0)

    if sq_velocity is not None:
        new_sq_velocity = beta2 * sq_velocity + (1 - beta2) * step ** 2
        new_sq_velocity = new_sq_velocity.masked_fill(inf_mask, 0.0)
        update = new_velocity / (new_sq_velocity.sqrt() + eps)
        update = update.masked_fill(inf_mask, 0.0)
        new_logits = (logits_f + update).masked_fill(inf_mask, float('-inf')).to(logits.dtype)
        return new_logits, new_velocity, new_sq_velocity

    new_logits = (logits_f + new_velocity).masked_fill(inf_mask, float('-inf')).to(logits.dtype)
    return new_logits, new_velocity


# ---------------------------------------------------------------------------
# PIVOTRolloutProcessor — one instance per vLLM generation request
# ---------------------------------------------------------------------------

class PIVOTRolloutProcessor:
    """
    vLLM logits_processor that applies Langevin sampling at temporal walk
    pivot points.

    Instantiate once per generation request and pass in
    `SamplingParams(logits_processors=[instance])`.

    Parameters
    ----------
    block_size : int
        Number of response tokens per causal block (default 8).
    threshold : float
        ||ΔR_t|| threshold to trigger Langevin (default 0.3).
    langevin_K : int
        Number of Langevin steps per trigger (default 3).
    langevin_eta : float
        Step size for entropy gradient (default 0.1).
    langevin_sigma : float
        Noise magnitude (default 0.01).
    """

    def __init__(
        self,
        block_size: int = 8,
        threshold: float = 0.3,
        langevin_K: int = 3,
        langevin_eta: float = 0.1,
        langevin_sigma: float = 0.01,
    ):
        self.block_size = block_size
        self.threshold = threshold
        self.K = langevin_K
        self.eta = langevin_eta
        self.sigma = langevin_sigma

        # Per-sequence state, maintained across decode steps
        self._k_history: List[torch.Tensor] = []  # (dim,) per past response tok
        self._C: Optional[torch.Tensor] = None     # (num_blocks, num_blocks) walk
        self._step: int = 0

        # Instrumentation: accumulated per-request
        self._trigger_count: int = 0
        self._entropy_before: List[float] = []
        self._entropy_after: List[float] = []
        # First-trigger example: top-5 token ids/probs before and after Langevin
        self._example: Optional[Dict] = None

    # ------------------------------------------------------------------
    # vLLM logits_processor interface
    # ------------------------------------------------------------------

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """
        Called by vLLM after each decode step for this sequence.

        token_ids : full list of token IDs generated so far (prompt + response)
        logits    : 1-D tensor of shape (vocab_size,) on whatever device vLLM uses
        """
        state = _pivot_decode_state

        # Claim this sequence's batch index
        with state.lock:
            seq_idx = state.cursor
            state.cursor += 1

        if state.q_batch is None or seq_idx >= state.q_batch.shape[0]:
            # Decode state not ready (e.g. prefill step, or state cleared)
            return logits

        # Retrieve Q/K for this sequence (on CPU to avoid device fragmentation)
        q_new = state.q_batch[seq_idx].float().cpu()   # (q_dim,)
        k_new = state.k_batch[seq_idx].float().cpu()   # (k_dim,)

        # Append current step's key to our shadow history
        self._k_history.append(k_new)
        self._step += 1

        # Need at least one full block to compute walk
        if len(self._k_history) < self.block_size:
            return logits

        # ----------------------------------------------------------------
        # Block-level attention: Q_new → past blocks
        # ----------------------------------------------------------------
        num_past = len(self._k_history)
        num_blocks = num_past // self.block_size  # complete blocks only
        if num_blocks == 0:
            return logits

        K_blocks = self._build_block_keys(num_blocks)  # (num_blocks, k_dim)

        # Q might have different dim than K due to GQA — project K_dim to Q_dim
        # by truncating or averaging heads to a shared head_dim
        k_dim = K_blocks.shape[-1]
        q_dim = q_new.shape[-1]
        if q_dim != k_dim:
            # Head averaging: assume both are num_heads * head_dim packed
            # Find GCD as head_dim proxy — use min-dim average
            if q_dim > k_dim:
                # q has more heads; average groups of q_dim//k_dim
                factor = q_dim // k_dim
                q_cmp = q_new.reshape(-1, factor).mean(-1)  # (k_dim,)
            else:
                factor = k_dim // q_dim
                K_blocks = K_blocks.reshape(num_blocks, -1, factor).mean(-1)  # (blocks, q_dim)
                q_cmp = q_new
        else:
            q_cmp = q_new  # (dim,)

        scale = q_cmp.shape[-1] ** 0.5
        scores = (q_cmp @ K_blocks.T) / scale  # (num_blocks,)
        a_t = torch.softmax(scores, dim=-1)     # attention over blocks

        # ----------------------------------------------------------------
        # Temporal walk update: R_t accumulates via C
        # ----------------------------------------------------------------
        t = num_blocks - 1  # current block index

        # Expand C if needed (new block arrived)
        if self._C is None or self._C.shape[0] < num_blocks:
            new_C = torch.zeros(num_blocks, num_blocks)
            if self._C is not None:
                n = self._C.shape[0]
                new_C[:n, :n] = self._C
            self._C = new_C

        if t > 0:
            multi = a_t[:t] @ self._C[:t, :t]   # (t,) multi-hop walk reach
            delta_norm = multi.norm().item()
            self._C[t, :t] = a_t[:t] + multi    # update walk row
        else:
            delta_norm = 0.0

        # ----------------------------------------------------------------
        # Langevin trigger
        # ----------------------------------------------------------------
        if delta_norm > self.threshold:
            self._trigger_count += 1

            with torch.no_grad():
                p_before = torch.softmax(logits.float(), dim=-1)
                H_before = -(p_before * torch.log(p_before.clamp(min=1e-12))).sum().item()

            # Capture top-5 example at first trigger per request
            if self._example is None:
                topk_before = torch.topk(p_before, 5)
                self._example = {
                    "step": self._step,
                    "delta_norm": delta_norm,
                    "topk_ids_before": topk_before.indices.tolist(),
                    "topk_probs_before": topk_before.values.tolist(),
                }

            logits = _langevin_sample(logits, self.K, self.eta, self.sigma)

            with torch.no_grad():
                p_after = torch.softmax(logits.float(), dim=-1)
                H_after = -(p_after * torch.log(p_after.clamp(min=1e-12))).sum().item()

            self._entropy_before.append(H_before)
            self._entropy_after.append(H_after)

            # Complete the example with after tokens (only once)
            if "topk_ids_after" not in self._example:
                topk_after = torch.topk(p_after, 5)
                self._example["topk_ids_after"] = topk_after.indices.tolist()
                self._example["topk_probs_after"] = topk_after.values.tolist()

        return logits

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_block_keys(self, num_blocks: int) -> torch.Tensor:
        """Average K_history into block-level keys: (num_blocks, k_dim)."""
        blocks = []
        for b in range(num_blocks):
            chunk = self._k_history[b * self.block_size:(b + 1) * self.block_size]
            blocks.append(torch.stack(chunk).mean(0))
        return torch.stack(blocks)  # (num_blocks, k_dim)


# ---------------------------------------------------------------------------
# Attention layer patching
# ---------------------------------------------------------------------------

def _get_model_layers(model) -> Optional[list]:
    """Return the list of transformer decoder layers, or None if not found."""
    # Common architectures
    for attr_path in ["model.layers", "language_model.layers", "model.model.layers"]:
        obj = model
        found = True
        for attr in attr_path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                found = False
                break
        if found and hasattr(obj, "__len__"):
            return list(obj)
    return None


def _get_inner_attn(layer) -> Optional[torch.nn.Module]:
    """Return the vllm.attention.Attention (inner) object from a decoder layer."""
    attn_module = getattr(layer, "self_attn", None)
    if attn_module is None:
        return None
    # vLLM attention layers typically have .attn (the Attention backend object)
    inner = getattr(attn_module, "attn", None)
    return inner


def patch_attention_layers_pivot(model) -> int:
    """
    Register forward pre-hooks on all attention layers of `model` to capture
    Q/K for each decode step and write averaged values to `_pivot_decode_state`.

    Returns the number of layers successfully patched.
    """
    state = _pivot_decode_state
    layers = _get_model_layers(model)
    if layers is None:
        logger.warning("PIVOT: could not find transformer layers in model — skipping attention patch")
        return 0

    num_layers = len(layers)
    patched = 0

    # Shared accumulators (captured by closure), cleared after all layers fire
    _layer_q: List[torch.Tensor] = []
    _layer_k: List[torch.Tensor] = []
    _fires = [0]  # mutable counter

    def make_hook(layer_idx: int):
        def _hook(module, args):
            """Pre-hook: fires before Attention.forward(query, key, value, ...)."""
            if len(args) < 2:
                return
            q = args[0].detach().float()  # (total_decode_tokens, q_heads * head_dim)
            k = args[1].detach().float()  # (total_decode_tokens, kv_heads * head_dim)

            with state.lock:
                _layer_q.append(q)
                _layer_k.append(k)
                _fires[0] += 1

                if _fires[0] == state.num_layers:
                    # All layers contributed — compute layer average
                    try:
                        state.q_batch = torch.stack(_layer_q).mean(0)
                        state.k_batch = torch.stack(_layer_k).mean(0)
                    except RuntimeError:
                        # Shape mismatch across layers (e.g. MLA) — use last layer
                        state.q_batch = _layer_q[-1]
                        state.k_batch = _layer_k[-1]
                    state.cursor = 0
                    _layer_q.clear()
                    _layer_k.clear()
                    _fires[0] = 0
        return _hook

    for layer_idx, layer in enumerate(layers):
        inner_attn = _get_inner_attn(layer)
        if inner_attn is None:
            logger.debug("PIVOT: no inner attn at layer %d, skipping", layer_idx)
            continue
        inner_attn.register_forward_pre_hook(make_hook(layer_idx))
        patched += 1

    with state.lock:
        state.num_layers = patched

    logger.info("PIVOT: patched %d / %d attention layers", patched, num_layers)
    return patched


# ---------------------------------------------------------------------------
# Factory: build a PIVOTRolloutProcessor from a config dict
# ---------------------------------------------------------------------------

def make_pivot_processor(pivot_cfg: dict) -> PIVOTRolloutProcessor:
    """
    Construct a PIVOTRolloutProcessor from a config dict.

    Expected keys (all optional, defaults shown):
      block_size       (int, 8)
      langevin_threshold (float, 0.3)
      langevin_K       (int, 3)
      langevin_eta     (float, 0.1)
      langevin_sigma   (float, 0.01)
    """
    return PIVOTRolloutProcessor(
        block_size=pivot_cfg.get("block_size", 8),
        threshold=pivot_cfg.get("langevin_threshold", 0.3),
        langevin_K=pivot_cfg.get("langevin_K", 3),
        langevin_eta=pivot_cfg.get("langevin_eta", 0.1),
        langevin_sigma=pivot_cfg.get("langevin_sigma", 0.01),
    )


# ---------------------------------------------------------------------------
# vLLM V1 engine-level adapter (registered at engine init, not per-request)
# ---------------------------------------------------------------------------

# Module-level config store — set by vllm_async_server.py before engine creation
# so the class constructor can read it without needing a custom __init__ signature.
_LANGEVIN_CFG: dict = {}

# Trigger registry removed: pivot_delta_vars (fork mask for entropy gating) is now
# computed on the actor side from response token IDs during update_policy, which
# avoids all cross-process IPC between vLLM worker subprocesses and Ray actors.


def set_langevin_cfg(cfg: dict) -> None:
    """Store Langevin config so PIVOTLangevinAdapter can read it at __init__.

    Also writes to an env var so that vLLM worker subprocesses (spawned after
    this call) inherit the config — the module-level _LANGEVIN_CFG is not
    propagated across spawn boundaries.
    """
    import json
    import os
    global _LANGEVIN_CFG
    serialisable = {k: v for k, v in cfg.items() if k != "tokenizer"}
    _LANGEVIN_CFG = dict(cfg)
    os.environ["_PIVOT_LANGEVIN_CFG"] = json.dumps(serialisable)


try:
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor as _AdapterLP

    class PIVOTLangevinAdapter(_AdapterLP):
        """
        vLLM V1 engine-level logits processor for PIVOT Phase 2.

        Register this class (not an instance) via::

            vllm_config.model_config.logits_processors = [PIVOTLangevinAdapter]

        vLLM will instantiate it with ``(vllm_config, device, is_pin_memory)``
        and call ``apply(logits)`` after every decode forward pass.

        Each request receives its own ``PIVOTRolloutProcessor`` (created by
        ``new_req_logits_processor``).  ``apply()`` is overridden to iterate
        in sorted batch-index order so that the cursor in
        ``_pivot_decode_state`` aligns with the attention hook's batch layout.
        """

        def __init__(self, vllm_config, device, is_pin_memory):
            super().__init__(vllm_config, device, is_pin_memory)
            self._pivot_cfg = dict(_LANGEVIN_CFG)

        def new_req_logits_processor(self, params):
            return PIVOTRolloutProcessor(
                block_size=self._pivot_cfg.get("block_size", 8),
                threshold=self._pivot_cfg.get("langevin_threshold", 0.3),
                langevin_K=self._pivot_cfg.get("langevin_K", 3),
                langevin_eta=self._pivot_cfg.get("langevin_eta", 0.1),
                langevin_sigma=self._pivot_cfg.get("langevin_sigma", 0.01),
            )

        def is_argmax_invariant(self) -> bool:
            return False

        def update_state(self, batch_update: Any) -> None:
            """Log per-request Langevin stats when requests leave the batch."""
            if batch_update and batch_update.removed:
                tokenizer = self._pivot_cfg.get("tokenizer", None)
                for req_idx in batch_update.removed:
                    if req_idx in self.req_info:
                        partial_fn = self.req_info[req_idx]
                        proc = getattr(partial_fn, "func", None)
                        if isinstance(proc, PIVOTRolloutProcessor) and proc._trigger_count > 0:
                            self._log_request_stats(req_idx, proc, tokenizer)
            super().update_state(batch_update)

        def _log_request_stats(
            self,
            req_idx: int,
            proc: "PIVOTRolloutProcessor",
            tokenizer: Any,
        ) -> None:
            n_triggers = proc._trigger_count
            n_steps = proc._step
            if n_steps == 0:
                return
            trigger_frac = n_triggers / n_steps
            H_before_mean = sum(proc._entropy_before) / n_triggers
            H_after_mean = sum(proc._entropy_after) / n_triggers
            delta_H = H_after_mean - H_before_mean
            logger.info(
                "PIVOT Langevin | req=%d steps=%d triggers=%d (%.1f%%) | "
                "H: before=%.3f after=%.3f Δ=%.3f",
                req_idx, n_steps, n_triggers, trigger_frac * 100,
                H_before_mean, H_after_mean, delta_H,
            )
            ex = proc._example
            if ex is not None and "topk_ids_after" in ex:
                if tokenizer is not None:
                    try:
                        before_strs = [tokenizer.decode([tid]) for tid in ex["topk_ids_before"]]
                        after_strs = [tokenizer.decode([tid]) for tid in ex["topk_ids_after"]]
                        before_fmt = " | ".join(
                            f"{repr(t)}({p:.3f})"
                            for t, p in zip(before_strs, ex["topk_probs_before"])
                        )
                        after_fmt = " | ".join(
                            f"{repr(t)}({p:.3f})"
                            for t, p in zip(after_strs, ex["topk_probs_after"])
                        )
                        logger.info(
                            "PIVOT Langevin example | step=%d δ=%.3f | "
                            "before=[%s] | after=[%s]",
                            ex["step"], ex["delta_norm"], before_fmt, after_fmt,
                        )
                    except Exception:
                        pass

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            """Override to iterate in sorted batch-index order.

            The attention hook resets ``_pivot_decode_state.cursor`` to 0 after
            the last layer fires, then each ``PIVOTRolloutProcessor`` call
            increments it.  Sorting by req_idx ensures the cursor increment
            matches the physical batch position written by the hook.
            """
            if self.req_info:
                for req_idx, req_lp in sorted(self.req_info.items()):
                    req_logits = logits[req_idx]
                    new_logits = req_lp(req_logits)
                    if new_logits is not req_logits:
                        logits[req_idx] = new_logits
            return logits

except ImportError:
    # vLLM V0 or AdapterLogitsProcessor not available — V1 adapter unavailable.
    PIVOTLangevinAdapter = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# PIVOT-v2: fork-profile-driven Langevin adapter
# ---------------------------------------------------------------------------

class PIVOTv2RolloutProcessor:
    """
    PIVOT-v2 per-request logits processor.

    Trigger priority:
      1. ΔVar[t] — mean variance of last-layer hidden states across n concurrent
         rollouts for the same prompt, set by PIVOTv2LangevinAdapter.apply()
         before each call.  Available from step 1 onward during training rollout
         (n > 1).  Fires when ``delta_var > delta_var_threshold``.
      2. Entropy fallback — fires when ``H[P(·|context)] > entropy_threshold``.
         Used at step 0 (prompt hash not yet set), at inference (n=1), and any
         time the adapter cannot supply a ΔVar value.

    After training, the model's entropy at fork positions is calibrated to be
    higher (co-design hypothesis), so the entropy fallback works at inference
    without needing concurrent rollouts.
    """

    def __init__(
        self,
        delta_var_threshold: float = 1.0,
        entropy_threshold: float = 2.0,
        langevin_K: int = 1,
        langevin_eta: float = 0.1,
        langevin_sigma: float = 0.01,
        langevin_top_k: int = 0,
        pivot_version: int = 2,
        adaptive_noise: bool = False,
        entropy_trigger_only: bool = False,
        langevin_momentum: float = 0.0,
        langevin_momentum_beta2: float = 0.0,
        langevin_mala: bool = False,
        langevin_min_trigger_position: int = 0,
        langevin_feedback: bool = False,
        langevin_exploit_ratio: float = 0.5,
        langevin_alpha_target: float = 0.0,
        langevin_eos_ids_tensor: Optional[torch.Tensor] = None,
    ):
        self.delta_var_threshold = delta_var_threshold
        self.entropy_threshold = entropy_threshold
        self.K = langevin_K
        self.eta = langevin_eta
        self.sigma = langevin_sigma
        self.top_k = langevin_top_k
        self.pivot_version = pivot_version
        self.adaptive_noise = adaptive_noise
        self.entropy_trigger_only = entropy_trigger_only
        self.langevin_momentum = langevin_momentum
        self.langevin_momentum_beta2 = langevin_momentum_beta2
        self.langevin_mala = langevin_mala
        self.langevin_min_trigger_position = langevin_min_trigger_position
        self.langevin_feedback = langevin_feedback
        self.langevin_exploit_ratio = langevin_exploit_ratio  # α: G-direction fraction in ε
        self.langevin_alpha_target = langevin_alpha_target    # v18: 0 = off, >0 = H_first*alpha target
        # v19: EOS-class token ids masked from the Langevin top-K subspace.
        # None / empty = mask off (legacy behavior).
        self.langevin_eos_ids_tensor = langevin_eos_ids_tensor

        # Set by PIVOTv2LangevinAdapter.apply() before each __call__
        self._delta_var_current: float = 0.0
        # v4: group mean logits set by adapter for group-mean perturbation
        self._group_mean_logits: Optional[torch.Tensor] = None
        # Set at step 0 from token_ids; used by adapter to group concurrent rollouts
        self.prompt_hash: Optional[int] = None

        self._step: int = 0
        self._trigger_count: int = 0
        self._trigger_by_delta_var: int = 0
        self._trigger_by_entropy: int = 0
        self._entropy_before: list = []
        self._entropy_after: list = []
        self._entropy_all: list = []    # every 4th position for calibration
        self._delta_vars_seen: list = []  # every 4th position for calibration
        self._example = None
        self._recent_non_trigger: list = []  # rolling 5-entry context buffer
        # Per-token delta_var for loss-phase entropy gating.
        # Appended each step: actual delta_var if delta_var trigger fired, else 0.0.
        self._step_delta_vars: list[float] = []
        # Per-trigger top-k log(p_lan): list of (topk_ids, topk_logp) tuples,
        # one per triggered step. Used to pre-compute IS denominator scalars in
        # update_state so dp_actor.py can skip the (n, n_trig, vocab) logit tensor.
        self._trigger_topk_logp: list = []  # list of (list[int], list[float]) — CPU
        self._lan_log_p_accum: list = []    # per-step log_p_lan cache (0.0 unresolved)
        # MALA accept/reject counters (only meaningful when langevin_mala=True).
        self._mala_accepted: int = 0
        self._mala_total: int = 0
        # Momentum velocity and second moment — persist across trigger positions.
        # None until the first trigger fires; naturally reset between sequences since
        # PIVOTv2RolloutProcessor is instantiated once per generation request.
        self._velocity: Optional[torch.Tensor] = None
        self._sq_velocity: Optional[torch.Tensor] = None
        # Feedback-guided Langevin state (langevin_feedback=True only).
        # G: accumulated gradient estimate direction (full-vocab, cpu).
        # _prev_eps: noise vector applied at last triggered position (for G update).
        # _fb_triggered: True when we're waiting for next-step H to update G.
        self._G: Optional[torch.Tensor] = None
        self._sq_G: Optional[torch.Tensor] = None   # Adam-RMS second moment for G
        self._prev_eps: Optional[torch.Tensor] = None
        self._fb_triggered: bool = False
        self._fb_signals: list = []     # per-trigger signal values for logging
        self._G_norm_last: float = 0.0  # G norm after last update
        # Live reference to the vLLM output token list for this request.
        # Attached by PIVOTv2LangevinAdapter.update_state when request starts.
        # By completion, this list contains the full generated sequence.
        self._output_tok_ids_ref: Optional[list] = None

    def _do_G_update(self, H_before: float) -> None:
        """Update G momentum from previous trigger's eps and current H_before.

        Called by the batched pre-pass in PIVOTv2LangevinAdapter.apply() before
        the batched Langevin step so that _G is up-to-date when used there.
        Sets _G_updated=True so __call__() skips the duplicate update.
        """
        if self.langevin_feedback and self._fb_triggered and self._prev_eps is not None:
            if self._entropy_before:
                if self.langevin_alpha_target > 0.0:
                    H_first = self._entropy_before[0]
                    H_target = H_first * self.langevin_alpha_target
                    signal = H_before - H_target
                else:
                    _buf = self._entropy_before[-50:]
                    _ref = sorted(_buf)[len(_buf) // 2]
                    signal = _ref - H_before
                self._fb_signals.append(signal)
                feedback = signal * self._prev_eps
                if self.langevin_momentum_beta2 > 0.0:
                    if self._sq_G is None:
                        self._sq_G = torch.zeros_like(feedback)
                    self._sq_G = (
                        self.langevin_momentum_beta2 * self._sq_G
                        + (1.0 - self.langevin_momentum_beta2) * feedback ** 2
                    )
                    feedback = feedback / (self._sq_G.sqrt() + 1e-8)
                gamma = self.langevin_momentum if self.langevin_momentum > 0.0 else 0.7
                if self._G is None:
                    self._G = (1.0 - gamma) * feedback
                else:
                    self._G = gamma * self._G + (1.0 - gamma) * feedback
            self._fb_triggered = False
        self._G_updated = True

    def __call__(self, prompt_ids, output_ids, logits: torch.Tensor) -> torch.Tensor:
        step = self._step
        self._step += 1

        # NaN sanitization is now handled in the adapter's batched pre-compute
        # pass (PIVOTv2LangevinAdapter.apply) before per-request processors run.

        # G update intentionally removed from here — see trigger block below.

        H = None  # set by trigger-determination block; reused in non-trigger logging

        # Set prompt hash from prompt_ids (constant per request, set by vLLM)
        if self.prompt_hash is None and prompt_ids is not None:
            try:
                self.prompt_hash = hash(tuple(int(t) for t in prompt_ids))
            except Exception:
                pass

        # Determine trigger: entropy-only mode, ΔVar, or entropy fallback.
        # Use pre-computed entropy from the batched adapter call when available
        # (avoids a per-request .item() CPU-GPU sync on every token).
        _precomputed_H = getattr(self, '_precomputed_H', None)
        self._precomputed_H = None  # consume immediately

        delta_var = 0.0  # default; overwritten in ΔVar branch
        if self.entropy_trigger_only:
            if _precomputed_H is not None:
                H = _precomputed_H
            else:
                with torch.no_grad():
                    p = torch.softmax(logits.float(), dim=-1)
                    H = -(p * torch.log(p.clamp(min=1e-12))).sum().item()
            if step % 4 == 0 and step >= self.langevin_min_trigger_position:
                self._entropy_all.append(H)
            fire = H > self.entropy_threshold
            trigger_mode = "entropy"
            _step_trig_val = H  # tentative; zeroed below if position guard suppresses
        else:
            delta_var = self._delta_var_current
            if delta_var > 0.0:
                if step % 4 == 0 and step >= self.langevin_min_trigger_position:
                    self._delta_vars_seen.append(delta_var)
                fire = delta_var > self.delta_var_threshold
                trigger_mode = "delta_var"
            else:
                if _precomputed_H is not None:
                    H = _precomputed_H
                else:
                    with torch.no_grad():
                        p = torch.softmax(logits.float(), dim=-1)
                        H = -(p * torch.log(p.clamp(min=1e-12))).sum().item()
                if step % 4 == 0 and step >= self.langevin_min_trigger_position:
                    self._entropy_all.append(H)
                fire = H > self.entropy_threshold
                trigger_mode = "entropy"
            # Only delta_var-mode triggers recorded; entropy fallback is not a
            # group-level branch-point signal and should not gate the loss.
            _step_trig_val = self._delta_var_current if (fire and trigger_mode == "delta_var") else 0.0

        if fire and self.langevin_min_trigger_position > 0 and step < self.langevin_min_trigger_position:
            fire = False

        # Append AFTER the position guard so the mask only reflects triggers that
        # actually fired (i.e. Langevin perturbation was applied). Recording pre-guard
        # fire caused fake trigger entries when resp_len < langevin_min_trigger_position,
        # leading to spurious IS corrections in the actor with log_p_lan=0.
        self._step_delta_vars.append(_step_trig_val if fire else 0.0)

        if not fire:
            if self._example is None:
                with torch.no_grad():
                    p = torch.softmax(logits.float(), dim=-1)
                    topk = torch.topk(p, 5)
                # Reuse H from trigger-determination (avoids .item() sync).
                H_nt = H if H is not None else float(
                    -(p * torch.log(p.clamp(min=1e-12))).sum().item()
                )
                self._recent_non_trigger.append({
                    "step": step, "topk_ids": topk.indices.tolist(),
                    "topk_probs": topk.values.tolist(), "H": H_nt,
                })
                if len(self._recent_non_trigger) > 5:
                    self._recent_non_trigger.pop(0)
            return logits

        self._trigger_count += 1
        if trigger_mode == "delta_var":
            self._trigger_by_delta_var += 1
        else:
            self._trigger_by_entropy += 1

        _precomputed_p = getattr(self, '_precomputed_p', None)
        self._precomputed_p = None
        with torch.no_grad():
            if _precomputed_p is not None:
                p_before = _precomputed_p
                H_before = _precomputed_H if _precomputed_H is not None else \
                    -(p_before * torch.log(p_before.clamp(min=1e-12))).sum().item()
            else:
                p_before = torch.softmax(logits.float(), dim=-1)
                H_before = -(p_before * torch.log(p_before.clamp(min=1e-12))).sum().item()

        # G update: fires at the next trigger (not t+1). H_before of this trigger
        # is the measurement used to update G momentum.
        # v17c: signal = median(H_past) - H_current  (always positive → always commits)
        # v18:  signal = H_current - H_first*alpha   (positive → commit, zero → stop,
        #                                             negative → reverse/explore)
        if getattr(self, '_G_updated', False):
            # Batched pre-pass in apply() already ran _do_G_update(H_before).
            self._G_updated = False
        elif self.langevin_feedback and self._fb_triggered and self._prev_eps is not None:
            if self._entropy_before:
                if self.langevin_alpha_target > 0.0:
                    # v18: target-band signal. H_first is the entropy at the first trigger
                    # this trajectory — model-agnostic reference that adapts to policy state.
                    H_first = self._entropy_before[0]
                    H_target = H_first * self.langevin_alpha_target
                    signal = H_before - H_target
                else:
                    # v17c: median-based signal
                    _buf = self._entropy_before[-50:]
                    _ref = sorted(_buf)[len(_buf) // 2]
                    signal = _ref - H_before
                self._fb_signals.append(signal)
                feedback = signal * self._prev_eps  # _prev_eps stays on GPU
                if self.langevin_momentum_beta2 > 0.0:
                    if self._sq_G is None:
                        self._sq_G = torch.zeros_like(feedback)
                    self._sq_G = (
                        self.langevin_momentum_beta2 * self._sq_G
                        + (1.0 - self.langevin_momentum_beta2) * feedback ** 2
                    )
                    feedback = feedback / (self._sq_G.sqrt() + 1e-8)
                gamma = self.langevin_momentum if self.langevin_momentum > 0.0 else 0.7
                if self._G is None:
                    self._G = (1.0 - gamma) * feedback
                else:
                    self._G = gamma * self._G + (1.0 - gamma) * feedback
                # Keep G on GPU — avoids 600 KB CPU↔GPU round-trip per trigger.
            self._fb_triggered = False

        if self._example is None:
            topk = torch.topk(p_before, 5)
            self._example = {
                "step": step, "trigger_mode": trigger_mode,
                "delta_var": delta_var, "H_before": H_before,
                "topk_ids_before": topk.indices.tolist(),
                "topk_probs_before": topk.values.tolist(),
                "non_trigger_context": list(self._recent_non_trigger),
            }

        if self.langevin_feedback:
            # v15i: adaptive G-guided noise, no entropy-gradient drift.
            _precomputed_eps = getattr(self, '_precomputed_eps', None)
            self._precomputed_eps = None  # consume
            if _precomputed_eps is not None:
                # Batched pre-pass already modified logits[req_idx] in-place and
                # computed eps. logits here IS the post-Langevin slice — just record.
                _eps = _precomputed_eps
            else:
                with torch.no_grad():
                    logits, _eps = _langevin_step_feedback(
                        logits, self.sigma, self.top_k, self._G, self.langevin_exploit_ratio,
                        eos_ids_tensor=self.langevin_eos_ids_tensor,
                    )
            self._prev_eps = _eps  # keep on GPU alongside _G
            self._fb_triggered = True
        elif self.pivot_version == 4 and self._group_mean_logits is not None and not self.langevin_mala:
            # v4: pull each rollout toward the group mean at this branch point.
            # logits_new = (1 - eta) * logits + eta * group_mean + noise_scale * eps
            # v4b (adaptive_noise): noise_scale = sigma * delta_var_norm, where
            #   delta_var_norm = delta_var / mean(|group_mean|)
            # This makes noise temperature proportional to inter-rollout divergence.
            with torch.no_grad():
                group_mean = self._group_mean_logits.to(logits.device, logits.dtype)
                finite_mask = torch.isfinite(logits)
                delta = self.eta * (group_mean - logits.float()).to(logits.dtype)
                if self.sigma > 0:
                    if self.adaptive_noise:
                        mean_abs = group_mean.abs().mean().item()
                        delta_var_norm = self._delta_var_current / (mean_abs + 1e-8)
                        noise_scale = self.sigma * delta_var_norm
                    else:
                        noise_scale = self.sigma
                    noise = (noise_scale * torch.randn_like(logits)).to(logits.dtype)
                else:
                    noise = torch.zeros_like(logits)
                if self.langevin_momentum > 0.0:
                    # Momentum: accumulate delta+noise into velocity across triggers.
                    if self._velocity is None:
                        self._velocity = torch.zeros_like(logits.float())
                    step = (delta + noise).float()
                    self._velocity = self.langevin_momentum * self._velocity + step
                    self._velocity = torch.where(finite_mask, self._velocity, torch.zeros_like(self._velocity))
                    logits = torch.where(finite_mask, logits + self._velocity.to(logits.dtype), logits)
                else:
                    logits = torch.where(finite_mask, logits + delta + noise, logits)
        else:
            if self.langevin_mala:
                # MALA: apply top-k mask first (restricts to d=top_k space for
                # good acceptance rate), then Metropolis-adjusted Langevin steps.
                if self.top_k > 0:
                    safe_logits = logits.float().nan_to_num(nan=float('-inf'))
                    topk_vals, topk_idx = torch.topk(safe_logits, min(self.top_k, logits.size(-1)))
                    masked = torch.full_like(safe_logits, float('-inf'))
                    masked.scatter_(-1, topk_idx, topk_vals)
                    logits = masked.to(logits.dtype)
                with torch.no_grad():
                    for _ in range(self.K):
                        logits, _accepted = _mala_step(logits, self.eta, self.sigma)
                        self._mala_total += 1
                        if _accepted:
                            self._mala_accepted += 1
            elif self.langevin_momentum > 0.0:
                # Momentum: velocity persists across trigger positions in this sequence.
                if self._velocity is None:
                    self._velocity = torch.zeros_like(logits.float())
                if self.langevin_momentum_beta2 > 0.0 and self._sq_velocity is None:
                    self._sq_velocity = torch.zeros_like(logits.float())
                with torch.no_grad():
                    result = _langevin_step_momentum(
                        logits, self.eta, self.sigma, self.langevin_momentum,
                        self._velocity, sq_velocity=self._sq_velocity,
                        beta2=self.langevin_momentum_beta2 if self.langevin_momentum_beta2 > 0.0 else 0.999,
                        top_k=self.top_k,
                    )
                    if self._sq_velocity is not None:
                        logits, self._velocity, self._sq_velocity = result
                    else:
                        logits, self._velocity = result
            else:
                logits = _langevin_sample(
                    logits, self.K, self.eta, self.sigma, top_k=self.top_k,
                    eos_ids_tensor=self.langevin_eos_ids_tensor,
                )

        with torch.no_grad():
            p_after = torch.softmax(logits.float(), dim=-1)
            # Keep H_after as a GPU tensor — defer .item() sync to update_state()
            # (called once per completed request, not once per decode step).
            H_after_t = -(p_after * torch.log(p_after.clamp(min=1e-12))).sum()
            # Store GPU tensors for topk log(p_lan) — defer .cpu().tolist() to
            # update_state() so we avoid N_trig CPU round-trips per decode step.
            _lp = torch.log_softmax(logits.float(), dim=-1)
            _fin = torch.isfinite(_lp)
            _topk_ids = _fin.nonzero(as_tuple=True)[0]
            self._trigger_topk_logp.append(
                (_topk_ids.cpu().tolist(), _lp[_topk_ids].cpu().tolist())
            )

        self._entropy_before.append(H_before)
        self._entropy_after.append(H_after_t)  # GPU tensor; converted in update_state()

        if self._example and "topk_ids_after" not in self._example:
            topk_after = torch.topk(p_after, 5)
            self._example["topk_ids_after"] = topk_after.indices.tolist()
            self._example["topk_probs_after"] = topk_after.values.tolist()

        return logits


try:
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor as _AdapterLP2

    class PIVOTv2LangevinAdapter(_AdapterLP2):
        """
        PIVOT-v2 engine-level logits processor.

        Trigger: per-prompt ΔVar[t] = mean variance of last-layer hidden states
        across the n concurrent rollouts for the same prompt, computed in
        apply() before each per-request processor is called.  Falls back to
        entropy threshold when only one rollout is active (inference n=1) or
        before prompt hashes are established (step 0).

        Register this class (not an instance) via::

            vllm_config.model_config.logits_processors = [PIVOTv2LangevinAdapter]

        The internalization KL loss (PIVOT-v6) is computed entirely on the actor
        side during update_policy, not here.  No trigger data needs to be
        transported from the rollout process to the training process.
        """

        def __init__(self, vllm_config, device, is_pin_memory):
            super().__init__(vllm_config, device, is_pin_memory)
            # Prefer env var (set by set_langevin_cfg before workers are spawned)
            # so config reaches worker subprocesses correctly across spawn boundaries.
            import json, os
            env_cfg = os.environ.get("_PIVOT_LANGEVIN_CFG", "")
            if env_cfg:
                self._pivot_cfg = json.loads(env_cfg)
            else:
                self._pivot_cfg = dict(_LANGEVIN_CFG)
            self._sample_logged: bool = False
            self._tokenizer = self._pivot_cfg.get("tokenizer", None)
            # v19: EOS-class token mask for Langevin top-K subspace.
            # Token ids are computed in the parent process (where the tokenizer
            # lives) and serialised through the env-var cfg as a list of ints;
            # we just rebuild the tensor here.  Shared by reference across all
            # per-request processors; never modified.
            self._eos_ids_tensor: Optional[torch.Tensor] = None
            if self._pivot_cfg.get("langevin_mask_eos", False):
                _eos_ids = self._pivot_cfg.get("langevin_eos_token_ids", [])
                if not _eos_ids and self._tokenizer is not None:
                    # Fallback: derive from tokenizer if the parent process didn't
                    # pre-compute (e.g. unit-test paths that set cfg directly).
                    _eos_ids = _identify_eos_class_tokens(self._tokenizer)
                if _eos_ids:
                    self._eos_ids_tensor = torch.tensor(
                        [int(t) for t in _eos_ids], dtype=torch.long,
                    )
                    logger.info(
                        "PIVOT v19: EOS-class mask enabled with %d token ids: %s",
                        len(_eos_ids), list(self._eos_ids_tensor.tolist()),
                    )
                else:
                    logger.warning(
                        "PIVOT v19: langevin_mask_eos=True but no eos token ids available; "
                        "mask will be a no-op.",
                    )
            # Aggregate stats
            self._agg_H_before: list = []
            self._agg_H_after: list = []
            self._agg_trigger_fracs: list = []
            self._agg_n_completed: int = 0
            self._agg_entropy_all: list = []
            self._agg_delta_vars: list = []
            self._agg_mala_accepted: int = 0
            self._agg_mala_total: int = 0
            self._agg_fb_signals: list = []  # feedback signal values across all requests
            # v19: rolling response-length buffer for adaptive t_min.
            # Appended in update_state() when each request completes; consumed in
            # new_req_logits_processor() to derive a quantile-based trigger floor.
            self._agg_response_lengths: list = []
            # Cross-process trigger registry via Manager IPC.
            # VllmHttpServer pickles self._shared_trig (the DictProxy pointing
            # to dict_A in the Manager server) into _PIVOT_TRIG_REGISTRY_PROXY.
            # Use the token ID from the env var to reconstruct a proxy to
            # dict_A (same object the parent holds), without calling
            # get_registry() again which would create a separate dict_B.
            _trig_addr_env = os.environ.get("_PIVOT_TRIG_REGISTRY_ADDR", "")
            if _trig_addr_env:
                try:
                    from multiprocessing.managers import Token, MakeProxyType
                    _cfg = json.loads(_trig_addr_env)
                    # address is either a Unix socket path (str) or [host, port]
                    _raw_addr = _cfg["address"]
                    _addr = tuple(_raw_addr) if isinstance(_raw_addr, list) else _raw_addr
                    _tok = Token(
                        typeid="get_registry",
                        address=_addr,
                        id=_cfg["proxy_id"],
                    )
                    _proxy_cls = MakeProxyType("DictProxy", _cfg["exposed"])
                    self._registry = _proxy_cls(
                        _tok, "pickle", authkey=_cfg["authkey"].encode("latin-1")
                    )
                except Exception as _e:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "PIVOT: failed to connect to trigger registry: %s; "
                        "trigger masks will not reach the actor.", _e)
                    self._registry = None
            else:
                self._registry = None

            # _slot_to_reqid maps batch slot index to request_id (from
            # SamplingParams.extra_args["_pivot_req_id"]).  Used as registry key
            # so we can write from apply() BEFORE output delivery (avoiding the
            # race where update_state fires at step N+1 but generate() pops at step N).
            self._slot_to_reqid: dict = {}
            # Keep for backward-compat; no longer written in apply() path.
            self._pending_writes: dict = {}

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        @staticmethod
        def _get_proc(req_lp) -> Optional["PIVOTv2RolloutProcessor"]:
            if isinstance(req_lp, PIVOTv2RolloutProcessor):
                return req_lp
            f = getattr(req_lp, "func", None)
            if isinstance(f, PIVOTv2RolloutProcessor):
                return f
            return None

        def new_req_logits_processor(self, params):
            # Adaptive threshold: if trig_percentile is set, compute the threshold
            # from the running buffer so the top (100-trig_percentile)% of positions
            # always trigger.  In entropy_trigger_only mode, uses the entropy buffer;
            # otherwise uses the delta_var buffer.
            #
            # trig_warmup_threshold: used during cold-start before buffer is large
            # enough for a reliable percentile estimate.  Set this high (e.g. 9.0)
            # so the first few requests barely trigger — the adaptive threshold
            # takes over after the first ~50 entropy samples are collected (≈1 req).
            # Defaults to entropy_threshold if not specified.
            trig_percentile = self._pivot_cfg.get("trig_percentile", None)
            entropy_trigger_only = self._pivot_cfg.get("entropy_trigger_only", False)

            fixed_dv_threshold = self._pivot_cfg.get("delta_var_threshold", 1.0)
            fixed_ent_threshold = self._pivot_cfg.get("entropy_threshold", 2.0)
            warmup_ent_threshold = self._pivot_cfg.get("trig_warmup_threshold", fixed_ent_threshold)
            # Use carry-forward threshold from previous step's calibration (set by per-step reset)
            warmup_ent_threshold = getattr(self, "_warmup_ent_threshold", warmup_ent_threshold)

            _min_buf = 50  # adaptive kicks in after ~50 entropy samples ≈ first request

            if entropy_trigger_only:
                if trig_percentile is not None and len(self._agg_entropy_all) >= _min_buf:
                    buf = self._agg_entropy_all[-2000:]
                    buf_sorted = sorted(buf)
                    idx = int(len(buf_sorted) * trig_percentile / 100.0)
                    idx = max(0, min(idx, len(buf_sorted) - 1))
                    computed_ent_threshold = buf_sorted[idx]
                    if not getattr(self, "_adaptive_ent_logged", False):
                        self._adaptive_ent_logged = True
                        n = len(buf_sorted)
                        frac_above = sum(1 for h in buf_sorted if h > computed_ent_threshold) / n
                        print(
                            f"PIVOT-v2 adaptive entropy threshold activated: "
                            f"p{trig_percentile}={computed_ent_threshold:.3f} "
                            f"buf={len(self._agg_entropy_all)} frac_above={frac_above:.3f}",
                            flush=True,
                        )
                else:
                    computed_ent_threshold = warmup_ent_threshold
                computed_dv_threshold = fixed_dv_threshold
            else:
                if trig_percentile is not None and len(self._agg_delta_vars) >= _min_buf:
                    buf = self._agg_delta_vars[-2000:]
                    buf_sorted = sorted(buf)
                    idx = int(len(buf_sorted) * trig_percentile / 100.0)
                    idx = max(0, min(idx, len(buf_sorted) - 1))
                    computed_dv_threshold = buf_sorted[idx]
                else:
                    computed_dv_threshold = fixed_dv_threshold
                computed_ent_threshold = fixed_ent_threshold

            # Adaptive t_min modes:
            #   "static"   (or unset): use the static langevin_min_trigger_position.
            #   "adaptive" (v19):      t_min = clamp(median(lengths) - W_min, floor, cap).
            #   "peak"     (v20):      t_min = clamp(alpha * historic_peak, floor, cap),
            #                          where historic_peak is a *monotonic* high-water
            #                          mark of median rollout length.  Once Langevin
            #                          self-disables due to length compression (collapse
            #                          onset), the peak preserves the disable: t_min stays
            #                          at alpha * peak_before_collapse, above any post-
            #                          collapse length.  Re-engages automatically if/when
            #                          the policy recovers and length climbs back over
            #                          t_min.  Replicates the manual "pump t_min up at
            #                          collapse onset" intervention that rescued v18b 1.7B.
            _t_min_static = int(self._pivot_cfg.get("langevin_min_trigger_position", 0))
            _t_min_mode = str(self._pivot_cfg.get("langevin_min_trigger_position_mode", "static")).lower()
            if _t_min_mode == "adaptive":
                _t_min_window = int(self._pivot_cfg.get("langevin_t_min_window", 400))
                _t_min_floor = int(self._pivot_cfg.get("langevin_t_min_floor", 400))
                _t_min_cap = int(self._pivot_cfg.get("langevin_t_min_cap", 2500))
                # Need at least ~64 completed rollouts for a stable median.
                # Below that, fall back to the static floor.
                if len(self._agg_response_lengths) >= 64:
                    _buf = sorted(self._agg_response_lengths[-2000:])
                    _median = int(_buf[len(_buf) // 2])
                    _t_min_computed = _median - _t_min_window
                    _t_min_effective = max(_t_min_floor, min(_t_min_computed, _t_min_cap))
                    if not getattr(self, "_adaptive_tmin_logged", False):
                        self._adaptive_tmin_logged = True
                        print(
                            f"PIVOT v19 adaptive t_min activated: "
                            f"median={_median} W_min={_t_min_window} -> t_min={_t_min_effective} "
                            f"(floor={_t_min_floor}, cap={_t_min_cap}, "
                            f"buf={len(self._agg_response_lengths)})",
                            flush=True,
                        )
                else:
                    _t_min_effective = _t_min_floor
            elif _t_min_mode == "peak":
                _alpha = float(self._pivot_cfg.get("langevin_peak_alpha", 0.6))
                _t_min_floor = int(self._pivot_cfg.get("langevin_t_min_floor", 200))
                _t_min_cap = int(self._pivot_cfg.get("langevin_t_min_cap", 3200))
                # Need ~64 completed rollouts before the peak is meaningful.
                if len(self._agg_response_lengths) >= 64:
                    _buf = sorted(self._agg_response_lengths[-2000:])
                    _median = int(_buf[len(_buf) // 2])
                    # Monotonic peak: only goes up, never down.  Once Langevin
                    # has fired on a long rollout regime, the peak preserves that
                    # information so any subsequent length collapse disables Langevin.
                    _prev_peak = getattr(self, "_peak_length", _t_min_floor)
                    self._peak_length = max(_prev_peak, _median)
                    _t_min_computed = int(_alpha * self._peak_length)
                    _t_min_effective = max(_t_min_floor, min(_t_min_computed, _t_min_cap))
                    # Log significant peak updates (first activation + each new peak).
                    if self._peak_length > _prev_peak + 50 or not getattr(self, "_peak_tmin_logged", False):
                        self._peak_tmin_logged = True
                        print(
                            f"PIVOT v20 peak t_min: "
                            f"median={_median} peak={self._peak_length} alpha={_alpha} "
                            f"-> t_min={_t_min_effective} "
                            f"(floor={_t_min_floor}, cap={_t_min_cap}, "
                            f"buf={len(self._agg_response_lengths)})",
                            flush=True,
                        )
                else:
                    _t_min_effective = _t_min_floor
            else:
                _t_min_effective = _t_min_static

            return PIVOTv2RolloutProcessor(
                delta_var_threshold=computed_dv_threshold,
                entropy_threshold=computed_ent_threshold,
                langevin_K=self._pivot_cfg.get("langevin_K", 1),
                langevin_eta=self._pivot_cfg.get("langevin_eta", 0.1),
                langevin_sigma=self._pivot_cfg.get("langevin_sigma", 0.01),
                langevin_top_k=self._pivot_cfg.get("langevin_top_k", 0),
                pivot_version=self._pivot_cfg.get("pivot_version", 2),
                adaptive_noise=self._pivot_cfg.get("adaptive_noise", False),
                entropy_trigger_only=self._pivot_cfg.get("entropy_trigger_only", False),
                langevin_momentum=float(self._pivot_cfg.get("langevin_momentum", 0.0)),
                langevin_momentum_beta2=float(self._pivot_cfg.get("langevin_momentum_beta2", 0.0)),
                langevin_mala=bool(self._pivot_cfg.get("langevin_mala", False)),
                langevin_min_trigger_position=_t_min_effective,
                langevin_feedback=bool(self._pivot_cfg.get("langevin_feedback", False)),
                langevin_exploit_ratio=float(self._pivot_cfg.get("langevin_exploit_ratio", 0.5)),
                langevin_alpha_target=float(self._pivot_cfg.get("langevin_alpha_target", 0.0)),
                langevin_eos_ids_tensor=self._eos_ids_tensor,
            )

        def is_argmax_invariant(self) -> bool:
            return False

        # ------------------------------------------------------------------
        # Core: compute ΔVar per prompt group, then apply per-request Langevin
        # ------------------------------------------------------------------

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            sorted_reqs = sorted(self.req_info.items())

            # ΔVar is only needed when NOT in entropy_trigger_only mode.
            # When entropy_trigger_only=True each processor ignores _delta_var_current
            # entirely, so computing it wastes ~128 .item() syncs per decode step.
            _entropy_trigger_only = self._pivot_cfg.get("entropy_trigger_only", False)
            if not _entropy_trigger_only:
                # Compute ΔVar[t] from logit distributions across concurrent rollouts.
                # Groups requests sharing the same prompt_hash (n=8 rollouts per prompt).
                # At step 0 all rollouts share the same logits (identical prefix) so var=0
                # and the entropy fallback handles the first token.  From step 1 onward,
                # diverged rollouts produce different logit vectors and var > 0.
                #
                # Using raw logit variance (pre-softmax) avoids hooks entirely and is
                # CUDA-graph compatible — no Python forward hooks needed.
                groups: dict = {}
                for req_idx, req_lp in sorted_reqs:
                    proc = self._get_proc(req_lp)
                    if proc is not None and proc.prompt_hash is not None:
                        groups.setdefault(proc.prompt_hash, []).append(req_idx)

                req_delta_vars: dict = {}
                _use_group_mean = int(self._pivot_cfg.get("pivot_version", 2)) == 4
                group_mean_logits: dict = {}  # prompt_hash -> mean logit tensor (v4 only)
                with torch.no_grad():
                    for ph, req_idxs in groups.items():
                        if len(req_idxs) >= 2:
                            logit_group = torch.stack(
                                [logits[idx].float() for idx in req_idxs]
                            )  # (n, vocab)
                            # Masked vocab positions have -inf across ALL rollouts; their
                            # cross-rollout variance is 0 by definition but -inf - (-inf)
                            # = nan.  Replace non-finite entries with 0 before computing.
                            logit_group = torch.where(
                                torch.isfinite(logit_group),
                                logit_group,
                                torch.zeros_like(logit_group),
                            )
                            mean_l = logit_group.mean(0)  # (vocab,) — needed for variance
                            var = ((logit_group - mean_l.unsqueeze(0)) ** 2).mean().item()
                            if _use_group_mean:
                                group_mean_logits[ph] = mean_l
                        else:
                            var = 0.0
                        for idx in req_idxs:
                            req_delta_vars[idx] = var

                # Inject ΔVar and group mean into each processor before calling it
                for req_idx, req_lp in sorted_reqs:
                    proc = self._get_proc(req_lp)
                    if proc is not None:
                        proc._delta_var_current = req_delta_vars.get(req_idx, 0.0)
                        ph = proc.prompt_hash
                        proc._group_mean_logits = group_mean_logits.get(ph, None) if (ph is not None and _use_group_mean) else None

            # Batched NaN sanitization + entropy pre-computation for all active
            # sequences in one pass — reduces CPU-GPU syncs from N_seq per token
            # step to 1 (for entropy tolist) + 0 (NaN: in-place, no sync).
            _H_list = None
            try:
                with torch.no_grad():
                    _idx_t = torch.tensor(
                        [idx for idx, _ in sorted_reqs],
                        dtype=torch.long, device=logits.device,
                    )
                    _active = logits[_idx_t].float()
                    # Sanitize NaN in-place on the gathered slice — nan_to_num is a
                    # Unconditional nan_to_num — no GPU sync (pure CUDA kernel).
                    # Replaces isfinite().all() which synced GPU→CPU every decode step.
                    _active.nan_to_num_(nan=float('-inf'))
                    _p_all = torch.softmax(_active, dim=-1)
                    _H_list = (-(_p_all * torch.log(_p_all.clamp(min=1e-12))).sum(dim=-1)).tolist()
                _p_list = _p_all.unbind(0)  # list of (vocab,) views, no copy
                for (_, req_lp), H_pre, p_pre in zip(sorted_reqs, _H_list, _p_list):
                    proc = self._get_proc(req_lp)
                    if proc is not None:
                        proc._precomputed_H = H_pre
                        proc._precomputed_p = p_pre  # reused at trigger to skip softmax
            except Exception:
                pass  # fall back to per-request .item() computation

            # Batched Langevin pre-pass for langevin_feedback=True sequences.
            # Replaces N_trig serial _langevin_step_feedback calls with one batched
            # call over (N_trig, vocab) — single randn + fused ops for all triggers.
            # Also runs _do_G_update() per triggered proc so G is current when used.
            _do_batched_lan = (
                self._pivot_cfg.get("langevin_feedback", False)
                and self._pivot_cfg.get("entropy_trigger_only", False)
                and _H_list  # pre-compute succeeded
            )
            if _do_batched_lan:
                try:
                    _trig_infos = []  # (sorted_idx, req_idx, proc, H)
                    for _bi, (req_idx, req_lp) in enumerate(sorted_reqs):
                        proc = self._get_proc(req_lp)
                        if proc is None or not proc.langevin_feedback:
                            continue
                        _H = _H_list[_bi]
                        _fire = (
                            _H > proc.entropy_threshold
                            and proc._step >= proc.langevin_min_trigger_position
                        )
                        if _fire:
                            proc._do_G_update(_H)
                            _trig_infos.append((_bi, req_idx, proc))

                    if _trig_infos:
                        _vocab = logits.shape[-1]
                        _n_trig = len(_trig_infos)
                        _trig_req_idxs = torch.tensor(
                            [req_idx for _, req_idx, _ in _trig_infos],
                            dtype=torch.long, device=logits.device,
                        )
                        _trig_logits = logits[_trig_req_idxs].float()  # (N_trig, vocab)

                        # Stack G tensors — zeros for procs with no G yet.
                        _G_stack = torch.zeros(_n_trig, _vocab, device=logits.device)
                        _proc0 = _trig_infos[0][2]
                        _sigma = _proc0.sigma
                        _top_k = _proc0.top_k
                        _exploit = _proc0.langevin_exploit_ratio
                        for _j, (_, _, _proc) in enumerate(_trig_infos):
                            if _proc._G is not None:
                                _G_stack[_j] = _proc._G.to(logits.device)

                        # One batched Langevin call for all N_trig sequences.
                        # All procs in the same batch share the same EOS mask tensor
                        # (built once per adapter instance from the tokenizer).
                        _eos_ids = getattr(_proc0, "langevin_eos_ids_tensor", None)
                        with torch.no_grad():
                            _new_trig_logits, _eps_batch = _langevin_step_feedback_batched(
                                _trig_logits, _sigma, _top_k, _G_stack, _exploit,
                                eos_ids_tensor=_eos_ids,
                            )

                        # Scatter first, inject eps second.  If scatter fails
                        # (OOM etc.), no eps are injected and __call__() falls back to
                        # per-request Langevin on the original logits — correct fallback.
                        # If scatter succeeds, all procs get eps before the loop runs.
                        logits[_trig_req_idxs] = _new_trig_logits.to(logits.dtype)
                        for _j, (_, _, _proc) in enumerate(_trig_infos):
                            _proc._precomputed_eps = _eps_batch[_j]
                except Exception:
                    pass  # fall back to per-request path on any error

            # Apply per-request logits processors (state bookkeeping for all procs;
            # triggered procs with _precomputed_eps set skip the Langevin GPU call).
            for req_idx, req_lp in sorted_reqs:
                req_logits = logits[req_idx]
                new_logits = req_lp(req_logits)
                if new_logits is not req_logits:
                    logits[req_idx] = new_logits

            self._flush_trigger_registry()
            return logits

        # ------------------------------------------------------------------
        # Registry flush (called from apply, before output delivery)
        # ------------------------------------------------------------------

        def _flush_trigger_registry(self) -> None:
            """Write one delta entry per triggered request this step.

            Key: (req_id, trig_idx) — O(1) constant size.
            Value: (step, dv, topk_ids, topk_lps) — ~20 ints + 20 floats per trigger.
            generate() collects (req_id, 0), (req_id, 1), ... and resolves log_p_lan
            from the final token_ids once generation is complete.
            """
            if self._registry is None:
                return
            _writes: dict = {}
            for req_idx, req_lp in self.req_info.items():
                proc = self._get_proc(req_lp)
                if proc is None or not proc._step_delta_vars:
                    continue
                if proc._step_delta_vars[-1] == 0.0:
                    continue
                _req_id = self._slot_to_reqid.get(req_idx)
                if _req_id is None:
                    continue
                _trig_idx = proc._trigger_count - 1
                _step = len(proc._step_delta_vars) - 1
                _dv = float(proc._step_delta_vars[-1])
                if _trig_idx < len(proc._trigger_topk_logp):
                    _ids, _lps = proc._trigger_topk_logp[_trig_idx]
                else:
                    _ids, _lps = [], []
                _writes[(_req_id, _trig_idx)] = (_step, _dv, _ids, _lps)
            if _writes:
                try:
                    self._registry.update(_writes)
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Stats logging
        # ------------------------------------------------------------------

        def update_state(self, batch_update) -> None:
            self._pending_writes = {}  # kept for compat; no longer used for writes
            if batch_update:
                # Capture the live output_tok_ids references for newly-added requests
                # BEFORE completion detection or super().update_state().  Each added
                # entry is (slot_idx, SamplingParams, prompt_tok_ids, output_tok_ids)
                # where output_tok_ids is a live Python list that grows during generation.
                # We store it on the processor after super() updates req_info with the
                # new processor, so future completions can write to _TRIGGER_REGISTRY.
                _new_req_tok_refs: dict[int, list] = {}
                # Also collect request_ids for new requests (for _slot_to_reqid).
                _new_slot_reqids: dict[int, str] = {}
                for added_entry in (batch_update.added or []):
                    try:
                        _new_req_tok_refs[added_entry[0]] = added_entry[3]
                        _params = added_entry[1]
                        _extra = getattr(_params, "extra_args", None) or {}
                        _req_id = _extra.get("_pivot_req_id")
                        if _req_id:
                            _new_slot_reqids[added_entry[0]] = _req_id
                    except (IndexError, TypeError, Exception):
                        pass

                # Collect moved slot pairs for _slot_to_reqid update.
                _moved_pairs: list = []
                for move_entry in (batch_update.moved or []):
                    try:
                        _moved_pairs.append((move_entry[0], move_entry[1]))
                    except (IndexError, TypeError):
                        pass

                # vLLM recycles freed slots immediately via pop_removed() in
                # _register_add_request, so batch_update.removed is almost
                # always empty by the time update_state is called.  Three paths
                # through which a completed request's processor can be found —
                # all must be checked BEFORE super().update_state() overwrites
                # req_info:
                #   1. removed: slot freed, not reused in this step
                #   2. added[i][0] in req_info: slot recycled by a new request
                #   3. moved[i][1] in req_info: slot filled by _condense moving
                #      an active request from the tail (most common path)
                completed_idxs: list[int] = list(batch_update.removed or [])
                for added_entry in (batch_update.added or []):
                    added_idx = added_entry[0]
                    if added_idx in self.req_info:
                        completed_idxs.append(added_idx)
                for move_entry in (batch_update.moved or []):
                    dst_idx = move_entry[1]
                    if dst_idx in self.req_info:
                        completed_idxs.append(dst_idx)
                for req_idx in completed_idxs:
                    if req_idx not in self.req_info:
                        continue
                    proc = self._get_proc(self.req_info[req_idx])
                    if not isinstance(proc, PIVOTv2RolloutProcessor):
                        continue
                    n_steps = proc._step
                    n_triggers = proc._trigger_count
                    if n_steps == 0:
                        continue
                    trigger_frac = n_triggers / n_steps
                    self._agg_trigger_fracs.append(trigger_frac)
                    # v19: response-length buffer for adaptive t_min.
                    # Cap retention at the most recent 4000 to bound memory.
                    self._agg_response_lengths.append(n_steps)
                    if len(self._agg_response_lengths) > 4000:
                        self._agg_response_lengths = self._agg_response_lengths[-2000:]
                    self._agg_n_completed += 1
                    self._agg_H_before.extend(proc._entropy_before)
                    # _entropy_after may contain GPU tensors (deferred from decode).
                    _H_after_floats = [float(h) for h in proc._entropy_after]
                    self._agg_H_after.extend(_H_after_floats)
                    if proc._entropy_all:
                        self._agg_entropy_all.extend(proc._entropy_all)
                    if proc._delta_vars_seen:
                        self._agg_delta_vars.extend(proc._delta_vars_seen)
                    self._agg_mala_accepted += proc._mala_accepted
                    self._agg_mala_total += proc._mala_total
                    if proc._fb_signals:
                        self._agg_fb_signals.extend(proc._fb_signals)

                    if n_triggers > 0:
                        H_b = sum(proc._entropy_before) / n_triggers if proc._entropy_before else 0.0
                        H_a = sum(_H_after_floats) / n_triggers if _H_after_floats else 0.0
                        fb_info = ""
                        if proc._fb_signals:
                            mean_sig = sum(proc._fb_signals) / len(proc._fb_signals)
                            pos_frac = sum(1 for s in proc._fb_signals if s > 0) / len(proc._fb_signals)
                            g_norm = float(proc._G.norm().item()) if proc._G is not None else 0.0
                            fb_info = (
                                f" | feedback: n={len(proc._fb_signals)} "
                                f"mean_signal={mean_sig:.4f} pos_frac={pos_frac:.2f} "
                                f"G_norm={g_norm:.4f}"
                            )
                        logger.info(
                            "PIVOT-v2 Langevin | req=%d steps=%d triggers=%d (%.1f%%) "
                            "[dv=%d ent=%d] | H_before=%.3f H_after=%.3f ΔH=%.3f%s",
                            req_idx, n_steps, n_triggers, trigger_frac * 100,
                            proc._trigger_by_delta_var, proc._trigger_by_entropy,
                            H_b, H_a, H_a - H_b, fb_info,
                        )
                        if not self._sample_logged and proc._example and "topk_ids_after" in proc._example:
                            self._sample_logged = True
                            self._print_trigger_sample(proc._example)

                # Aggregate summary every 64 completed requests (once per boundary crossing)
                _n64_now = self._agg_n_completed // 64
                if _n64_now > 0 and _n64_now > getattr(self, "_last_summary_n64", 0):
                    self._last_summary_n64 = _n64_now
                    recent_fracs = self._agg_trigger_fracs[-64:]
                    mean_frac = sum(recent_fracs) / len(recent_fracs)
                    if self._agg_H_before:
                        rH_b = self._agg_H_before[-64:]
                        rH_a = self._agg_H_after[-64:]
                        mH_b = sum(rH_b) / len(rH_b)
                        mH_a = sum(rH_a) / len(rH_a)
                        print(
                            f"PIVOT-v2 [summary] completed={self._agg_n_completed} "
                            f"trigger_frac={mean_frac:.3f} "
                            f"H_before={mH_b:.3f} H_after={mH_a:.3f} ΔH={mH_a - mH_b:.3f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"PIVOT-v2 [summary] completed={self._agg_n_completed} "
                            f"trigger_frac={mean_frac:.3f} no triggers",
                            flush=True,
                        )
                    if self._agg_fb_signals:
                        recent_sigs = self._agg_fb_signals[-512:]
                        mean_sig = sum(recent_sigs) / len(recent_sigs)
                        pos_frac = sum(1 for s in recent_sigs if s > 0) / len(recent_sigs)
                        neg_frac = 1.0 - pos_frac
                        print(
                            f"PIVOT-v2 [feedback] n={len(recent_sigs)} "
                            f"mean_signal={mean_sig:.4f} "
                            f"pos_frac={pos_frac:.2f} neg_frac={neg_frac:.2f} "
                            f"(+signal=entropy fell=constructive, -signal=entropy rose=disruptive)",
                            flush=True,
                        )
                    if self._agg_mala_total > 0:
                        accept_rate = self._agg_mala_accepted / self._agg_mala_total
                        print(
                            f"PIVOT-v2 [mala] proposals={self._agg_mala_total} "
                            f"accepted={self._agg_mala_accepted} "
                            f"accept_rate={accept_rate:.3f}",
                            flush=True,
                        )
                    # ΔVar distribution — calibration for delta_var_threshold
                    if self._agg_delta_vars:
                        dv_s = sorted(self._agg_delta_vars[-2000:])
                        n = len(dv_s)
                        dvth = self._pivot_cfg.get("delta_var_threshold", 1.0)
                        print(
                            f"PIVOT-v2 ΔVar dist | n={n} | "
                            f"p50={dv_s[n // 2]:.4f} p90={dv_s[int(n * 0.90)]:.4f} "
                            f"p95={dv_s[int(n * 0.95)]:.4f} p99={dv_s[int(n * 0.99)]:.4f} | "
                            f"threshold={dvth:.3f} frac_above={sum(1 for v in dv_s if v > dvth) / n:.3f}",
                            flush=True,
                        )
                    # Entropy distribution — calibration for entropy_threshold
                    if self._agg_entropy_all:
                        ent_s = sorted(self._agg_entropy_all[-2000:])
                        n = len(ent_s)
                        eth = self._pivot_cfg.get("entropy_threshold", 2.0)
                        trig_pct = self._pivot_cfg.get("trig_percentile", None)
                        if trig_pct is not None and n >= 50:
                            adaptive_idx = max(0, min(int(n * trig_pct / 100.0), n - 1))
                            adaptive_eth = ent_s[adaptive_idx]
                            target_frac = 1.0 - trig_pct / 100.0
                            actual_frac = sum(1 for h in ent_s if h > adaptive_eth) / n
                            print(
                                f"PIVOT-v2 entropy dist | n={n} | "
                                f"p50={ent_s[n // 2]:.3f} p80={ent_s[int(n * 0.80)]:.3f} "
                                f"p90={ent_s[int(n * 0.90)]:.3f} p95={ent_s[int(n * 0.95)]:.3f} | "
                                f"adaptive_threshold={adaptive_eth:.3f} target_frac={target_frac:.2f} actual_frac={actual_frac:.3f}",
                                flush=True,
                            )
                        else:
                            print(
                                f"PIVOT-v2 entropy dist | n={n} | "
                                f"p50={ent_s[n // 2]:.3f} p90={ent_s[int(n * 0.90)]:.3f} "
                                f"p95={ent_s[int(n * 0.95)]:.3f} p99={ent_s[int(n * 0.99)]:.3f} | "
                                f"threshold={eth:.1f} frac_above={sum(1 for h in ent_s if h > eth) / n:.3f}",
                                flush=True,
                            )
            else:
                _new_req_tok_refs = {}
                _new_slot_reqids = {}
                _moved_pairs = []
                completed_idxs = []
            # Per-step buffer reset: vLLMHttpServer writes "_pivot_reset" to the
            # registry proxy before each rollout so the adaptive threshold is
            # calibrated from the current step's entropy distribution only.
            if self._registry is not None:
                try:
                    if self._registry.pop("_pivot_reset", None):
                        # Save current step's calibrated threshold before clearing buffer
                        _trig_percentile = self._pivot_cfg.get("trig_percentile", None)
                        if _trig_percentile is not None and len(self._agg_entropy_all) >= 50:
                            _buf = sorted(self._agg_entropy_all[-2000:])
                            _idx = int(len(_buf) * _trig_percentile / 100.0)
                            _idx = max(0, min(_idx, len(_buf) - 1))
                            self._warmup_ent_threshold = _buf[_idx]
                        self._agg_entropy_all = []
                        self._agg_delta_vars = []
                        self._agg_mala_accepted = 0
                        self._agg_mala_total = 0
                        self._agg_fb_signals = []
                        self._adaptive_ent_logged = False
                except Exception:
                    pass

            # Update _slot_to_reqid: remove completed, update moved, add new.
            # This keeps apply()'s _flush_trigger_registry() able to look up
            # the request_id for each active slot.
            for req_idx in completed_idxs:
                self._slot_to_reqid.pop(req_idx, None)
            for src, dst in _moved_pairs:
                _rid = self._slot_to_reqid.pop(src, None)
                if _rid is not None:
                    self._slot_to_reqid[dst] = _rid
            for slot_idx, req_id in _new_slot_reqids.items():
                self._slot_to_reqid[slot_idx] = req_id

            super().update_state(batch_update)
            # Attach the live output_tok_ids reference to each newly-added processor.
            # super().update_state() added them to req_info; now we can look them up.
            for slot_idx, tok_ref in _new_req_tok_refs.items():
                if slot_idx in self.req_info:
                    proc = self._get_proc(self.req_info[slot_idx])
                    if isinstance(proc, PIVOTv2RolloutProcessor):
                        proc._output_tok_ids_ref = tok_ref

        def _print_trigger_sample(self, example: dict) -> None:
            tok = self._tokenizer
            def decode(ids): return [tok.decode([i]) if tok else str(i) for i in ids]
            def fmt_top5(ids, probs):
                return "  ".join(f"{repr(t):12s}({p:.3f})" for t, p in zip(decode(ids), probs))

            sep = "=" * 72
            mode = example.get("trigger_mode", "?")
            dv = example.get("delta_var", 0.0)
            H_b = example.get("H_before", 0.0)
            lines = [
                "", sep,
                f"[PIVOT-v2 Langevin] first trigger | mode={mode} ΔVar={dv:.4f} H_before={H_b:.3f}",
                sep,
            ]
            for c in example.get("non_trigger_context", []):
                lines.append(f"  pos={c['step']:4d}  H={c['H']:.3f}  {fmt_top5(c['topk_ids'], c['topk_probs'])}")
            lines.append(f"  pos={example['step']:4d}  BEFORE  {fmt_top5(example['topk_ids_before'], example['topk_probs_before'])}")
            if "topk_ids_after" in example:
                lines.append(f"  pos={example['step']:4d}  AFTER   {fmt_top5(example['topk_ids_after'], example['topk_probs_after'])}")
            lines.append(sep)
            print("\n".join(lines), flush=True)

    class PIVOTv18bLangevinAdapter(PIVOTv2LangevinAdapter):
        """PIVOT v18b: CUDA-graph-captured Langevin kernel with GPU-resident G state.

        Subclass of PIVOTv2LangevinAdapter — inherits IS correction, trigger
        registry, and all bookkeeping.  Replaces the batched Langevin pre-pass
        with a single CUDA graph replay over a (max_slots, vocab) static buffer.

        Key differences from v18:
        - G state lives fully on GPU (no per-proc CPU ↔ GPU transfers for G).
        - entropy_threshold is taken from config at init time; the adaptive
          trig_percentile path is not wired into the graph (fixed threshold).
        - One-step G phase: G is updated BEFORE graph replay so the Langevin
          step uses the latest G (same semantics as per-proc v18 path).

        Enable via: +actor_rollout_ref.rollout.pivot.lan_use_cuda_graph=True
        Falls back to PIVOTv2LangevinAdapter.apply() if CUDA is unavailable or
        graph capture fails.
        """

        def __init__(self, vllm_config, device, is_pin_memory):
            super().__init__(vllm_config, device, is_pin_memory)
            from collections import deque as _deque
            self._v18b_device = device
            max_slots = vllm_config.scheduler_config.max_num_seqs
            self._max_slots = max_slots
            self._slot_map: dict = {}
            self._free_slots: "_deque[int]" = _deque(range(max_slots))
            # GPU state (lazy init on first apply())
            self._V: Optional[int] = None
            self._G_state = None          # (max_slots, V) float32
            self._H_first_state = None    # (max_slots,) float32
            self._has_trig_state = None   # (max_slots,) bool
            self._prev_eps_state = None   # (max_slots, V) float32
            self._step_counts_state = None  # (max_slots,) int64
            # Graph input buffers
            self._graph_logits = None     # (max_slots, V) float32
            self._graph_noise = None      # (max_slots, V) float32
            # Graph output references (set during capture)
            self._graph_out_logits = None
            self._graph_trig_mask = None
            self._graph_eps_out = None
            self._graph_H = None
            self._cuda_graph = None

        def _v18b_init_gpu(self, V: int) -> None:
            device = self._v18b_device
            N = self._max_slots
            self._V = V
            self._G_state = torch.zeros(N, V, device=device, dtype=torch.float32)
            self._H_first_state = torch.zeros(N, device=device, dtype=torch.float32)
            self._has_trig_state = torch.zeros(N, device=device, dtype=torch.bool)
            self._prev_eps_state = torch.zeros(N, V, device=device, dtype=torch.float32)
            self._step_counts_state = torch.zeros(N, device=device, dtype=torch.int64)
            self._graph_logits = torch.zeros(N, V, device=device, dtype=torch.float32)
            self._graph_noise = torch.zeros(N, V, device=device, dtype=torch.float32)
            self._graph_ent_thresh = torch.tensor(
                float(self._pivot_cfg.get("entropy_threshold", 0.4)),
                device=device, dtype=torch.float32,
            )
            self._v18b_capture_graph()

        def _v18b_graph_params(self) -> dict:
            cfg = self._pivot_cfg
            return dict(
                sigma=float(cfg.get("langevin_sigma", 0.01)),
                top_k=int(cfg.get("langevin_top_k", 20)),
                exploit_ratio=float(cfg.get("langevin_exploit_ratio", 0.6)),
                min_trigger_position=int(cfg.get("langevin_min_trigger_position", 200)),
                alpha_target=float(cfg.get("langevin_alpha_target", 0.0)),
            )

        def _v18b_capture_graph(self) -> None:
            params = self._v18b_graph_params()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._graph_noise.normal_()
                    _pivot_v18_graph_kernel(
                        self._graph_logits, self._graph_noise,
                        self._G_state, self._step_counts_state,
                        self._has_trig_state, self._H_first_state,
                        entropy_threshold=self._graph_ent_thresh,
                        **params,
                    )
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            self._graph_noise.normal_()
            with torch.cuda.graph(g):
                (self._graph_out_logits,
                 self._graph_trig_mask,
                 self._graph_eps_out,
                 self._graph_H) = _pivot_v18_graph_kernel(
                    self._graph_logits, self._graph_noise,
                    self._G_state, self._step_counts_state,
                    self._has_trig_state, self._H_first_state,
                    entropy_threshold=self._graph_ent_thresh,
                    **params,
                )
            self._cuda_graph = g

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            # Lazy GPU init on first call (vocab size now known)
            if self._V is None:
                try:
                    if not torch.cuda.is_available():
                        return super().apply(logits)
                    self._v18b_init_gpu(logits.shape[-1])
                except Exception:
                    return super().apply(logits)

            if self._cuda_graph is None:
                return super().apply(logits)

            sorted_reqs = sorted(self.req_info.items())
            device = logits.device

            # Assign slots for new requests; fall back if pool is exhausted
            for _, req_lp in sorted_reqs:
                if id(req_lp) not in self._slot_map:
                    if not self._free_slots:
                        return super().apply(logits)
                    self._slot_map[id(req_lp)] = self._free_slots.popleft()

            # Gather active (req_idx, slot_idx, proc) triples
            active_req_idxs: list[int] = []
            active_slot_idxs: list[int] = []
            active_procs = []
            for req_idx, req_lp in sorted_reqs:
                slot = self._slot_map.get(id(req_lp))
                proc = self._get_proc(req_lp)
                if slot is None or proc is None:
                    continue
                active_req_idxs.append(req_idx)
                active_slot_idxs.append(slot)
                active_procs.append(proc)

            if not active_req_idxs:
                return super().apply(logits)

            with torch.no_grad():
                req_idx_t = torch.tensor(active_req_idxs, dtype=torch.long, device=device)
                slot_idx_t = torch.tensor(active_slot_idxs, dtype=torch.long, device=device)

                # NaN sanitize + copy into graph input buffer
                active_f32 = logits[req_idx_t].float()
                active_f32.nan_to_num_(nan=float('-inf'))
                self._graph_logits[slot_idx_t] = active_f32

                # Update step counters for active slots
                steps_t = torch.tensor(
                    [p._step for p in active_procs], dtype=torch.int64, device=device
                )
                self._step_counts_state[slot_idx_t] = steps_t

                # Compute entropy BEFORE graph so G can be updated with correct H
                p_all = torch.softmax(active_f32, dim=-1)
                H_active = -(p_all * torch.log(p_all.clamp(min=1e-12))).sum(dim=-1)
                H_list = H_active.tolist()  # one sync per step

            # Determine which procs trigger — use one consistent adaptive threshold
            # (same value for graph replay and proc_triggers so IS mask matches)
            cfg = self._pivot_cfg
            alpha_target = float(cfg.get("langevin_alpha_target", 0.0))
            gamma = float(cfg.get("langevin_momentum", 0.7))
            min_trig_pos = int(cfg.get("langevin_min_trigger_position", 200))

            _cur_ent_thresh = float(active_procs[0].entropy_threshold)
            self._graph_ent_thresh.fill_(_cur_ent_thresh)

            proc_triggers = [
                H > _cur_ent_thresh and proc._step >= min_trig_pos
                for proc, H in zip(active_procs, H_list)
            ]

            # Update G BEFORE graph replay so Langevin uses the latest G
            if any(proc_triggers):
                with torch.no_grad():
                    trig_slots = [active_slot_idxs[i] for i, f in enumerate(proc_triggers) if f]
                    trig_slot_t = torch.tensor(trig_slots, dtype=torch.long, device=device)
                    trig_H_vals = torch.tensor(
                        [H_list[i] for i, f in enumerate(proc_triggers) if f],
                        dtype=torch.float32, device=device,
                    )
                    # Build full-size trigger mask and H for _pivot_v18_update_G
                    trig_mask_full = torch.zeros(
                        self._max_slots, dtype=torch.bool, device=device
                    )
                    trig_mask_full[trig_slot_t] = True
                    H_full = self._graph_H.clone()  # reuse graph buffer shape; values overwritten
                    H_full[trig_slot_t] = trig_H_vals
                    _pivot_v18_update_G(
                        self._G_state, self._H_first_state, self._has_trig_state,
                        self._prev_eps_state, trig_mask_full, H_full,
                        alpha_target=alpha_target, gamma=gamma,
                    )

            # Pre-fill noise outside graph, then replay
            self._graph_noise.normal_()
            self._cuda_graph.replay()

            # Scatter modified logits back for triggered rows + set precomputed fields
            trig_req_idxs = [active_req_idxs[i] for i, f in enumerate(proc_triggers) if f]
            trig_slot_idxs_list = [active_slot_idxs[i] for i, f in enumerate(proc_triggers) if f]

            if trig_req_idxs:
                with torch.no_grad():
                    trig_req_t = torch.tensor(trig_req_idxs, dtype=torch.long, device=device)
                    trig_slot_t2 = torch.tensor(trig_slot_idxs_list, dtype=torch.long, device=device)
                    logits[trig_req_t] = self._graph_out_logits[trig_slot_t2].to(logits.dtype)
                    # Update prev_eps for triggered slots
                    self._prev_eps_state[trig_slot_t2] = self._graph_eps_out[trig_slot_t2]

            # Inject precomputed H and eps into each proc before __call__()
            for i, proc in enumerate(active_procs):
                proc._precomputed_H = H_list[i]
                if proc_triggers[i]:
                    proc._precomputed_eps = self._graph_eps_out[active_slot_idxs[i]]
                    proc._G_updated = True  # suppress redundant G update in __call__()

            # Per-request bookkeeping loop (records trigger events, stats, etc.)
            for req_idx, req_lp in sorted_reqs:
                req_logits = logits[req_idx]
                new_logits = req_lp(req_logits)
                if new_logits is not req_logits:
                    logits[req_idx] = new_logits

            # Flush triggered requests' data to registry NOW (before output delivery).
            self._flush_trigger_registry()

            return logits

        def update_state(self, batch_update) -> None:
            # Collect completed req_lp objects BEFORE super() overwrites req_info
            completed_lps: list = []
            if batch_update:
                completed_idxs: list[int] = list(batch_update.removed or [])
                for added_entry in (batch_update.added or []):
                    if added_entry[0] in self.req_info:
                        completed_idxs.append(added_entry[0])
                for move_entry in (batch_update.moved or []):
                    if move_entry[1] in self.req_info:
                        completed_idxs.append(move_entry[1])
                for idx in completed_idxs:
                    if idx in self.req_info:
                        completed_lps.append(self.req_info[idx])

            super().update_state(batch_update)

            # Free slots and reset GPU state for completed requests
            for req_lp in completed_lps:
                key = id(req_lp)
                if key not in self._slot_map:
                    continue
                slot = self._slot_map.pop(key)
                if self._G_state is not None:
                    self._G_state[slot].zero_()
                    self._H_first_state[slot].zero_()
                    self._has_trig_state[slot] = False
                    self._prev_eps_state[slot].zero_()
                    self._step_counts_state[slot] = 0
                self._free_slots.append(slot)

except ImportError:
    PIVOTv2LangevinAdapter = None  # type: ignore[assignment,misc]
    PIVOTv18bLangevinAdapter = None  # type: ignore[assignment,misc]
