"""
Walk-based token importance scoring for W-GRPO advantage weighting.

During the actor's log-prob forward pass, forward hooks capture Q and K
projections at each attention layer. These are block-pooled and accumulated
into a cross-layer walk matrix R^k via repeated matrix multiplication.

The per-token importance score is a causally-masked, normalized column sum
of the layer-averaged walk state:

    w[j] = (1 / valid_count[j]) * Σ_{i >= j} R_avg[i, j]

Only rows i >= j contribute (causal: block i can only reference block j if j
came before or at i). Normalized by valid_count[j] = (num_blocks - j) so
that early and late blocks are comparable.

    Â_t = Â * (1 + alpha * w̃_block(t))
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class WalkImportanceComputer:
    """
    Computes layer-averaged walk importance scores from a model forward pass.

    Usage:
        computer = WalkImportanceComputer(block_size=32, walk_degree=4)
        handles, layer_qk = computer.register_hooks(model)
        # ... run model forward pass ...
        for h in handles: h.remove()
        scores = computer.compute(layer_qk, cu_seqlens, max_seqlen, B, response_mask)
    """

    def __init__(self, block_size: int = 32, walk_degree: int = 4,
                 norm_mode: str = "relu_max", agg_mode: str = "causal",
                 # legacy param — ignored, kept for config backward compat
                 hadamard_dim: int = 64):
        """
        Args:
            block_size:  Tokens per block for mean-pooling Q/K before walk.
            walk_degree: Number of walk steps per layer (R = R @ W repeated).
            norm_mode:
                "relu_max"     : ReLU(z-score) / max → [0, 1]. Below-average=0
                                 (no effect), above-average amplified.
                                 weight = 1 + α*w ∈ [1, 1+α]. (v2/v4/v5)
                "zscore_absmax": z-score / abs_max → [-1, 1]. Below-average
                                 suppressed, above-average amplified.
                                 weight ∈ [1-α, 1+α], clamped to [0, 1+α]. (v3)
            agg_mode:
                "causal"   : w[j] = mean_{i >= j} R_avg[i,j]. Only rows that
                             can causally attend to j contribute; normalized by
                             valid count so early/late blocks are comparable.
                             (v5 — default)
                "response" : w[j] = Σ_{i ∈ response blocks} R_avg[i,j].
                             Only response tokens act as queries. (v4)
                "all"      : w[j] = Σ_i R_avg[i,j]. All rows, no masking. (v2/v3)
        """
        self.block_size = block_size
        self.walk_degree = walk_degree
        self.norm_mode = norm_mode
        self.agg_mode = agg_mode
        # Set by register_hooks from model config
        self.num_q_heads: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0

    # ------------------------------------------------------------------
    # Block pooling
    # ------------------------------------------------------------------

    def _block_pool(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """Mean-pool (B, H, T, D) -> (B, H, num_blocks, D)."""
        if block_size <= 1:
            return x
        B, H, T, D = x.shape
        num_blocks = (T + block_size - 1) // block_size
        pad = num_blocks * block_size - T
        if pad:
            x = F.pad(x, (0, 0, 0, pad), value=0.0)
        return x.view(B, H, num_blocks, block_size, D).mean(dim=3)

    # ------------------------------------------------------------------
    # Block-level attention probability matrix
    # ------------------------------------------------------------------

    def _attn_probs_block(
        self,
        query_states: torch.Tensor,  # (B, Hq, T, D)
        key_states: torch.Tensor,    # (B, Hkv, T, D)
        block_size: int,
    ) -> torch.Tensor:
        """
        Compute block-level attention probability matrix via direct dot product
        on block-pooled Q and K. No Hadamard sketch needed — after block pooling
        the matrix is only (num_blocks × num_blocks), trivially small.

        Returns (B, num_blocks, num_blocks) row-stochastic matrix W where
        W[i,j] represents coupling strength from block i (query) to block j (key).
        """
        # Expand KV heads to match query heads (GQA / MQA support)
        if key_states.size(1) != query_states.size(1):
            num_groups = query_states.size(1) // key_states.size(1)
            key_states = key_states.repeat_interleave(num_groups, dim=1)

        # Block pool: (B, H, T, D) -> (B, H, num_blocks, D)
        q = self._block_pool(query_states, block_size)
        k = self._block_pool(key_states, block_size)

        # Cast to bfloat16 for efficiency
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)

        # Scaled dot-product: (B, H, num_blocks, num_blocks)
        scale = q.size(-1) ** -0.5
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # Softmax with clamping for numerical stability
        attn_probs = torch.softmax(logits.clamp(-10, 10), dim=-1)

        # Aggregate over heads → (B, num_blocks, num_blocks), re-normalize rows
        pt_blk = attn_probs.sum(dim=1)
        return pt_blk / (pt_blk.sum(dim=-1, keepdim=True) + 1e-12)

    # ------------------------------------------------------------------
    # Walk accumulation
    # ------------------------------------------------------------------

    def _accumulate_walk(
        self,
        attn_probs: torch.Tensor,          # (B, num_blocks, num_blocks)
        prev_walk: Optional[torch.Tensor],  # (B, num_blocks, num_blocks) or None
    ) -> torch.Tensor:
        """R^k = R^{k-1} @ W^k  (repeated walk_degree times).
        On first layer, R^0 = W^0."""
        W = attn_probs
        state = prev_walk if prev_walk is not None else W
        for _ in range(self.walk_degree):
            state = torch.bmm(state, W)
        return state

    # ------------------------------------------------------------------
    # Q/K hook registration
    # ------------------------------------------------------------------

    def register_hooks(self, model: torch.nn.Module):
        """
        Register output hooks on q_proj and k_proj of every attention layer.

        Returns:
            handles:   list of hook handles (call h.remove() after forward pass)
            layer_qk:  dict populated in-place during forward pass:
                       layer_qk[layer_idx] = {'q': tensor, 'k': tensor}
                       tensors are (total_tokens, H*D) in packed mode.
        """
        model_inner = model
        while hasattr(model_inner, "module"):
            model_inner = model_inner.module

        cfg = model_inner.config
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        layer_qk: dict[int, dict[str, torch.Tensor]] = {}
        handles = []
        layer_idx = [0]

        for name, module in model.named_modules():
            if not (hasattr(module, "q_proj") and hasattr(module, "k_proj")):
                continue
            has_child_attn = any(hasattr(child, "q_proj") for child in module.children())
            if has_child_attn:
                continue

            idx = layer_idx[0]
            layer_idx[0] += 1

            def make_q_hook(i):
                def hook(module, input, output):
                    if i not in layer_qk:
                        layer_qk[i] = {}
                    layer_qk[i]["q"] = output.detach()
                return hook

            def make_k_hook(i):
                def hook(module, input, output):
                    if i not in layer_qk:
                        layer_qk[i] = {}
                    layer_qk[i]["k"] = output.detach()
                return hook

            handles.append(module.q_proj.register_forward_hook(make_q_hook(idx)))
            handles.append(module.k_proj.register_forward_hook(make_k_hook(idx)))

        return handles, layer_qk

    # ------------------------------------------------------------------
    # Unpacking packed tensors
    # ------------------------------------------------------------------

    def _unpack_projection(
        self,
        packed: torch.Tensor,      # (total_tokens, num_heads * head_dim)
        num_heads: int,
        head_dim: int,
        cu_seqlens: torch.Tensor,  # (B+1,) int32
        max_seqlen: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Unpack packed (total_tokens, H*D) -> (B, H, max_seqlen, D)."""
        packed = packed.view(-1, num_heads, head_dim)
        result = torch.zeros(
            batch_size, num_heads, max_seqlen, head_dim,
            device=packed.device, dtype=packed.dtype,
        )
        for i in range(batch_size):
            s = cu_seqlens[i].item()
            e = cu_seqlens[i + 1].item()
            result[i, :, :e - s, :] = packed[s:e].transpose(0, 1)
        return result

    # ------------------------------------------------------------------
    # Token-level expansion
    # ------------------------------------------------------------------

    def _expand_blocks_to_tokens(
        self,
        w_block: torch.Tensor,        # (B, num_blocks)
        cu_seqlens: torch.Tensor,     # (B+1,) cumulative actual sequence lengths
        response_mask: torch.Tensor,  # (B, max_response_length)
    ) -> torch.Tensor:
        """
        Expand block-level scores to token level and extract response portion.
        Each token in block i gets score w_block[:, i].
        Returns (B, max_response_length).
        """
        B, num_blocks = w_block.shape
        max_response_length = response_mask.size(1)

        w_token_full = w_block.repeat_interleave(self.block_size, dim=1)

        output = torch.zeros(B, max_response_length, device=w_block.device, dtype=w_block.dtype)
        for i in range(B):
            seq_len_i = int((cu_seqlens[i + 1] - cu_seqlens[i]).item())
            resp_len_i = int(response_mask[i].sum().item())
            if resp_len_i == 0:
                continue
            tok_end = min(seq_len_i, w_token_full.size(1))
            tok_start = max(0, tok_end - resp_len_i)
            actual = tok_end - tok_start
            if actual > 0:
                output[i, :actual] = w_token_full[i, tok_start:tok_end]

        return output * response_mask.float()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute(
        self,
        layer_qk: dict[int, dict[str, torch.Tensor]],
        cu_seqlens: torch.Tensor,                      # (B+1,) int32
        max_seqlen: int,
        batch_size: int,
        response_mask: torch.Tensor,                   # (B, response_length)
        prompt_lens: Optional[torch.Tensor] = None,    # (B,) int32
    ) -> torch.Tensor:
        """
        Compute walk-based per-token importance scores for the response tokens.

        Returns:
            Tensor of shape (B, response_length), zero outside response tokens.
        """
        if not layer_qk:
            return torch.zeros(
                batch_size, response_mask.size(1),
                device=response_mask.device, dtype=torch.float32,
            )

        walk_states = []
        prev_walk = None

        for layer_idx in sorted(layer_qk.keys()):
            packed_q = layer_qk[layer_idx]["q"]  # (total_tokens, Hq*D)
            packed_k = layer_qk[layer_idx]["k"]  # (total_tokens, Hkv*D)

            q = self._unpack_projection(
                packed_q, self.num_q_heads, self.head_dim,
                cu_seqlens, max_seqlen, batch_size,
            )
            k = self._unpack_projection(
                packed_k, self.num_kv_heads, self.head_dim,
                cu_seqlens, max_seqlen, batch_size,
            )

            # Block-level attention matrix: (B, num_blocks, num_blocks)
            attn_probs = self._attn_probs_block(q, k, self.block_size)

            # Walk accumulation: R^k
            prev_walk = self._accumulate_walk(attn_probs, prev_walk)
            walk_states.append(prev_walk.clone())

        # Layer-averaged walk state: (B, num_blocks, num_blocks)
        R_avg = torch.stack(walk_states, dim=0).mean(dim=0).float()
        num_blocks = R_avg.size(1)

        # ------------------------------------------------------------------
        # Column sum: w[j] = how much walk mass arrives at block j
        # agg_mode controls which query rows contribute.
        # ------------------------------------------------------------------
        if self.agg_mode == "causal":
            # Only rows i >= j can causally attend to j.
            # Normalize by valid_count[j] = (num_blocks - j) for fair comparison.
            # Lower-triangular mask: causal_mask[i,j] = 1 iff i >= j
            causal_mask = torch.ones(num_blocks, num_blocks,
                                     device=R_avg.device).tril()  # (b, b)
            valid_counts = torch.arange(num_blocks, 0, -1,
                                        device=R_avg.device).float()  # [b, b-1, ..., 1]
            w_block = (R_avg * causal_mask.unsqueeze(0)).sum(dim=1)  # (B, num_blocks)
            w_block = w_block / valid_counts.unsqueeze(0)            # normalize

        elif self.agg_mode == "response" and prompt_lens is not None:
            # Only response query blocks contribute.
            resp_block_starts = (prompt_lens // self.block_size).long()
            row_idx = torch.arange(num_blocks, device=R_avg.device).unsqueeze(0)
            agg_mask = (row_idx >= resp_block_starts.unsqueeze(1)).float()
            w_block = (R_avg * agg_mask.unsqueeze(2)).sum(dim=1)

        else:
            # All rows contribute (legacy v2/v3 behaviour).
            w_block = R_avg.sum(dim=1)

        # ------------------------------------------------------------------
        # Normalize per sequence
        # ------------------------------------------------------------------
        w_mean = w_block.mean(dim=-1, keepdim=True)
        w_std = w_block.std(dim=-1, keepdim=True).clamp(min=1e-6)
        w_zscore = (w_block - w_mean) / w_std

        if self.norm_mode == "relu_max":
            w_relu = F.relu(w_zscore)
            w_max = w_relu.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
            w_block = w_relu / w_max
        elif self.norm_mode == "zscore_absmax":
            w_abs_max = w_zscore.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6)
            w_block = w_zscore / w_abs_max
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

        return self._expand_blocks_to_tokens(w_block, cu_seqlens, response_mask)


def apply_walk_weighted_advantage(
    advantages: torch.Tensor,       # (B, response_length)
    walk_importance: torch.Tensor,  # (B, response_length)
    response_mask: torch.Tensor,    # (B, response_length)
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Reweight GRPO advantages by walk importance scores.
    Â_t = Â * (1 + alpha * w̃_block(t))
    Setting alpha=0 recovers standard GRPO.
    """
    weight = (1.0 + alpha * walk_importance).clamp(min=0.0)
    return advantages * weight * response_mask


# ---------------------------------------------------------------------------
# PIVOT: Temporal Walk Computer
# ---------------------------------------------------------------------------

class TemporalWalkComputer(WalkImportanceComputer):
    """
    Computes per-token temporal walk change scores for PIVOT loss gating.

    Key difference from WalkImportanceComputer (layer-wise, depth-wise walk):
    Instead of accumulating the walk across transformer LAYERS for a single
    next-token prediction, this accumulates across TOKEN GENERATION STEPS,
    tracking how the multi-hop causal structure evolves over time.

    At each response block t:
        a_t   = layer-averaged attention row of block t  (B, num_blocks)
        multi = a_t @ C                                  (B, num_blocks) — multi-hop via C
        score = ||multi||_2                              (B,)
        C[t]  = a_t + multi                              update walk state

    High score_t → block t introduces new multi-hop causal pathways →
    structural change point → PIVOT Langevin trigger and gradient gate.

    The causal mask is explicitly applied to W before the temporal walk so
    that block t can only attend to blocks ≤ t, matching generation order.
    """

    @torch.no_grad()
    def compute(
        self,
        layer_qk: dict[int, dict[str, torch.Tensor]],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        batch_size: int,
        response_mask: torch.Tensor,
        prompt_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns (B, response_length) of normalised ||ΔR_t||_F scores.
        Zero outside response tokens; high values mark structural change points.
        """
        if not layer_qk:
            return torch.zeros(
                batch_size, response_mask.size(1),
                device=response_mask.device, dtype=torch.float32,
            )

        # ------------------------------------------------------------------
        # 1. Layer-averaged block-level attention matrix W
        # ------------------------------------------------------------------
        all_attn: list[torch.Tensor] = []
        for layer_idx in sorted(layer_qk.keys()):
            packed_q = layer_qk[layer_idx]["q"]
            packed_k = layer_qk[layer_idx]["k"]

            q = self._unpack_projection(
                packed_q, self.num_q_heads, self.head_dim,
                cu_seqlens, max_seqlen, batch_size,
            )
            k = self._unpack_projection(
                packed_k, self.num_kv_heads, self.head_dim,
                cu_seqlens, max_seqlen, batch_size,
            )
            all_attn.append(self._attn_probs_block(q, k, self.block_size))

        W = torch.stack(all_attn, dim=0).mean(dim=0).float()  # (B, num_blocks, num_blocks)
        num_blocks = W.size(1)

        # ------------------------------------------------------------------
        # 2. Apply causal mask: block t may only attend to blocks ≤ t
        # ------------------------------------------------------------------
        causal = torch.tril(torch.ones(num_blocks, num_blocks, device=W.device))
        W = W * causal.unsqueeze(0)
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-12)

        # ------------------------------------------------------------------
        # 3. Temporal walk: process blocks in generation order
        #    C[b, t, :] = walk row for block t (a_t + multi-hop via C)
        # ------------------------------------------------------------------
        C = torch.zeros(batch_size, num_blocks, num_blocks, device=W.device)
        delta_norms = torch.zeros(batch_size, num_blocks, device=W.device)

        for t in range(num_blocks):
            a_t = W[:, t, :]                                   # (B, num_blocks)
            multi = torch.bmm(a_t.unsqueeze(1), C).squeeze(1)  # (B, num_blocks)
            C[:, t, :] = a_t + multi
            delta_norms[:, t] = multi.norm(dim=-1)             # (B,)

        # ------------------------------------------------------------------
        # 4. Identify response blocks for normalisation
        # ------------------------------------------------------------------
        if prompt_lens is not None:
            prompt_block_counts = (prompt_lens.float() / self.block_size).ceil().long()
        else:
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
            resp_lens = response_mask.sum(dim=-1).long()
            prompt_block_counts = ((seq_lens - resp_lens).float() / self.block_size).ceil().long()

        block_idx = torch.arange(num_blocks, device=W.device).unsqueeze(0)   # (1, B)
        resp_mask_b = (block_idx >= prompt_block_counts.unsqueeze(1)).float() # (B, num_blocks)

        # ------------------------------------------------------------------
        # 5. Normalise within response blocks per sequence
        # ------------------------------------------------------------------
        resp_delta = delta_norms * resp_mask_b
        n_resp = resp_mask_b.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = resp_delta.sum(dim=-1, keepdim=True) / n_resp
        sq_mean = (resp_delta ** 2).sum(dim=-1, keepdim=True) / n_resp
        std = (sq_mean - mean ** 2).clamp(min=0.0).sqrt().clamp(min=1e-6)
        w_zscore = (resp_delta - mean) / std

        if self.norm_mode == "relu_max":
            w_relu = F.relu(w_zscore)
            w_max = w_relu.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
            w_block = (w_relu / w_max) * resp_mask_b
        elif self.norm_mode == "zscore_absmax":
            w_abs = w_zscore.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6)
            w_block = (w_zscore / w_abs) * resp_mask_b
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

        return self._expand_blocks_to_tokens(w_block, cu_seqlens, response_mask)


# ---------------------------------------------------------------------------
# PIVOT: loss weight helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PIVOT-v2: Representation Divergence Computer
# ---------------------------------------------------------------------------

class RepresentationDivergenceComputer:
    """
    Compute reward-conditioned representation divergence for PIVOT-v2 Phase 1.

    At each response position t, for each prompt group (n consecutive rollouts):
        v[t]          = mean(h_t | correct rollouts) - mean(h_t | wrong rollouts)
        fork_score[t] = ||v[t]||_2

    Scores are broadcast to all n sequences in the group, then normalised
    within each response using relu_max (zero for below-average positions,
    scaled to [0,1] above).

    When all rollouts in a group have the same reward sign (no_contrast),
    returns zeros for that group — falls back to standard GRPO.

    Difference from v1 temporal walk:
    - v1: per-rollout temporal walk (no reward conditioning)
    - v2: cross-rollout reward-conditioned divergence in representation space
    - v2 hooks last-layer hidden states (not Q/K across all layers)
    - v2 requires no extra forward pass when merged with log-prob computation
    """

    def __init__(self, norm_mode: str = "relu_max"):
        self.norm_mode = norm_mode

    def register_last_layer_hook(self, model: torch.nn.Module):
        """
        Register a forward output hook on the last transformer decoder layer.

        Returns:
            handle : hook handle (call handle.remove() after forward pass)
            store  : dict populated in-place with key 'h' during forward pass.
                     h shape: (total_tokens, hidden_dim) if remove_padding,
                               else (batch_size, seq_len, hidden_dim).
        """
        inner = model
        while hasattr(inner, "module"):
            inner = inner.module

        layers = None
        for path in ["model.layers", "language_model.model.layers", "transformer.h"]:
            obj = inner
            ok = True
            for attr in path.split("."):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__") and len(obj) > 0:
                layers = obj
                break

        if layers is None:
            return None, {}

        store: dict = {}

        def _hook(module, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            store["h"] = h.detach()

        handle = layers[-1].register_forward_hook(_hook)
        return handle, store

    @torch.no_grad()
    def compute_from_hidden(
        self,
        h_resp: torch.Tensor,           # (B, response_len, hidden_dim), any device
        response_mask: torch.Tensor,    # (B, response_len)
        rewards: torch.Tensor,          # (B,) scalar reward per sequence
        n_per_group: int,
    ) -> torch.Tensor:
        """
        Compute and normalise fork scores from pre-extracted response hidden states.

        Returns:
            (B, response_len) tensor on same device as h_resp.
        """
        B, response_len, _ = h_resp.shape
        device = h_resp.device
        fork_scores = torch.zeros(B, response_len, device=device, dtype=torch.float32)

        n_groups = B // n_per_group
        for g in range(n_groups):
            s, e = g * n_per_group, g * n_per_group + n_per_group
            group_h = h_resp[s:e].float()    # (n, response_len, hidden_dim)
            group_r = rewards[s:e]           # (n,)
            group_mask = response_mask[s:e]  # (n, response_len)

            correct = (group_r > 0)
            wrong = (group_r < 0)

            if correct.sum() == 0 or wrong.sum() == 0:
                # No contrast — zeros, falls back to standard GRPO
                continue

            mean_correct = group_h[correct].mean(dim=0)  # (response_len, hidden_dim)
            mean_wrong = group_h[wrong].mean(dim=0)      # (response_len, hidden_dim)

            v = mean_correct - mean_wrong                 # (response_len, hidden_dim)
            score = v.norm(dim=-1)                        # (response_len,)

            # Broadcast same fork score to all n sequences in the group
            for i in range(n_per_group):
                fork_scores[s + i] = score * group_mask[i].float()

        return self._normalise(fork_scores, response_mask)

    def _normalise(
        self,
        scores: torch.Tensor,        # (B, response_len)
        response_mask: torch.Tensor, # (B, response_len)
    ) -> torch.Tensor:
        rm = response_mask.float()
        scores = scores * rm

        if self.norm_mode != "relu_max":
            raise ValueError(f"Unknown norm_mode for RepresentationDivergenceComputer: {self.norm_mode!r}")

        n = rm.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (scores * rm).sum(dim=-1, keepdim=True) / n
        sq_mean = ((scores ** 2) * rm).sum(dim=-1, keepdim=True) / n
        std = (sq_mean - mean ** 2).clamp(min=0.0).sqrt().clamp(min=1e-6)
        z = (scores - mean) / std
        z = torch.relu(z)
        z_max = (z * rm).max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        return (z / z_max) * rm


def apply_pivot_loss_weights(
    pg_losses: torch.Tensor,        # (B, response_length) — per-token policy losses
    pivot_scores: torch.Tensor,     # (B, response_length) — normalised ||ΔR_t|| scores
    response_mask: torch.Tensor,    # (B, response_length)
    mode: str = "soft",
    threshold: float = 0.3,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Gate policy gradient losses by PIVOT temporal walk change scores.

    mode="soft":   pg_losses *= (1 + alpha * pivot_scores)   — continuous upweighting
    mode="binary": pg_losses *= (pivot_scores > threshold)   — hard mask at P_causal

    Returns (B, response_length) masked losses.
    """
    if mode == "soft":
        weight = (1.0 + alpha * pivot_scores).clamp(min=0.0)
    elif mode == "binary":
        weight = (pivot_scores > threshold).float()
    else:
        raise ValueError(f"Unknown PIVOT mode: {mode!r}. Use 'soft' or 'binary'.")
    return pg_losses * weight * response_mask
