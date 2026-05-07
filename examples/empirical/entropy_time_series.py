#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""
Branch-point entropy time-series within a trajectory.

For a fixed prompt, generate a single rollout under two conditions:
    (A) Vanilla:  no Langevin perturbation (sample from softmax(l_t)).
    (B) DRIFT:    Langevin perturbation at branch points, exactly as in
                  Algorithm 1 of the paper (entropy-maintenance feedback).

For both conditions, log per-token (position, entropy) at every position, plus
the trigger mask under condition (B).  The resulting time-series shows that
branch-point entropy under (B) tracks α_target · H_first while (A) decays to
near zero — the empirical signature of the entropy-maintenance signal.

This script uses HuggingFace transformers directly (no vLLM, no Ray) so we
have token-level control over each forward.  Run on a single GPU.

Usage:
    python entropy_time_series.py \
        --ckpt /storage/.../models/Qwen3-1.7B-Base \
        --prompts /storage/.../data/guru_rl/test_aime.parquet \
        --n_prompts 5 \
        --max_new_tokens 1500 \
        --tag drift_v18b_step120 \
        --langevin_eta 0.1 --langevin_sigma 0.01 --top_k 20 \
        --p_quantile 0.85 --t_min 800 --alpha_target 0.7 \
        --out_dir ./entropy_out

Reads prompts from a verl-format parquet (uses 'prompt' column), runs both
conditions on the same prompt, dumps a per-trajectory JSON for each.  Plot
H_t vs branch-point index from the resulting JSONL.

Output schema (one record per (prompt, condition)):
    {
      "tag": "drift_v18b_step120",
      "prompt_idx": 0,
      "condition": "drift" | "vanilla",
      "tokens": [...],
      "entropies": [...],          # full per-token H_t time-series
      "is_trigger": [0/1, ...],    # 1 at positions where Langevin fired (drift only)
      "H_first": float | null,     # entropy at the first trigger (drift only)
      "alpha_target": 0.7,
      "trigger_positions": [int, ...],
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def shannon_entropy(logits):
    """H of softmax(logits) in nats.  Logits: (V,)."""
    p = F.softmax(logits.float(), dim=-1)
    log_p = torch.log(p.clamp(min=1e-12))
    return float(-(p * log_p).sum().item())


def langevin_perturb_topk(logits, K, eta, sigma, eps_dir, generator=None):
    """One Langevin step on the top-K coordinates.

    logits: (V,)  raw logits, in-place safe (returns a fresh tensor).
    eta:     drift step size.
    sigma:   diffusion magnitude.
    eps_dir: (K,) unit perturbation direction in the rank-ordered top-K space.
    """
    new_logits = logits.clone()
    # Top-K selection by absolute logit
    topk_vals, topk_idx = torch.topk(logits.abs(), K)
    # Apply drift on those K coords
    drift = eta * eps_dir.to(logits.device, logits.dtype)
    new_logits[topk_idx] = new_logits[topk_idx] + drift
    # Add isotropic Gaussian noise to top-K only
    noise = torch.randn(K, generator=generator, device=logits.device, dtype=logits.dtype) * sigma
    new_logits[topk_idx] = new_logits[topk_idx] + noise
    return new_logits


def sample_unit_sphere(K, generator=None, device="cuda", dtype=torch.float32):
    v = torch.randn(K, generator=generator, device=device, dtype=dtype)
    return v / (v.norm() + 1e-12)


@torch.no_grad()
def rollout(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    K,
    eta,
    sigma,
    p_quantile,
    t_min,
    alpha_target,
    gamma,
    alpha,
    use_langevin,
    seed,
    buf_size=2000,
    warmup_threshold=0.4,
):
    """Generate a single rollout with optional Langevin perturbation, logging
    full per-token entropy and the trigger mask."""
    device = model.device
    gen = torch.Generator(device=device).manual_seed(seed)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Rolling entropy buffer (global, shared across this trajectory)
    h_buf = []

    entropies = []
    triggers = []
    sampled_tokens = []
    H_first = None
    G = torch.zeros(K, device=device, dtype=torch.float32)
    eps_prev = None
    trigger_positions = []

    # Use KV cache by feeding one token at a time after the initial forward
    past_key_values = None
    cur = input_ids
    for t in range(max_new_tokens):
        out = model(cur, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits_t = out.logits[0, -1, :]  # (V,)

        # Compute entropy of unperturbed distribution at t
        H_t = shannon_entropy(logits_t)
        entropies.append(H_t)

        # Threshold for branch-point selection
        if len(h_buf) < 50:
            tau_t = warmup_threshold
        else:
            sorted_buf = sorted(h_buf[-buf_size:])
            tau_t = sorted_buf[int(p_quantile * len(sorted_buf))]
        is_branch = use_langevin and H_t >= tau_t and t > t_min

        if is_branch:
            triggers.append(1)
            trigger_positions.append(t)
            if H_first is None:
                H_first = H_t

            # Update G with the previous trigger's eps using the
            # entropy-maintenance feedback signal s = H_t - alpha_target * H_first
            if eps_prev is not None:
                signal = H_t - alpha_target * H_first
                G = gamma * G + (1.0 - gamma) * signal * eps_prev

            # Mix learned drift with sphere exploration
            xi = sample_unit_sphere(K, generator=gen, device=device, dtype=torch.float32)
            if G.norm().item() > 1e-6:
                eps_t = alpha * (G / G.norm()) + (1.0 - alpha) * xi
            else:
                eps_t = xi

            # Apply Langevin update
            new_logits = langevin_perturb_topk(logits_t, K, eta, sigma, eps_t, generator=gen)
            probs = F.softmax(new_logits, dim=-1)
            eps_prev = eps_t
        else:
            triggers.append(0)
            probs = F.softmax(logits_t, dim=-1)

        # Sample next token (categorical with the active distribution)
        idx = torch.multinomial(probs, 1, generator=gen).item()
        sampled_tokens.append(idx)

        h_buf.append(H_t)
        if len(h_buf) > buf_size:
            h_buf = h_buf[-buf_size:]

        if idx == tokenizer.eos_token_id:
            break

        cur = torch.tensor([[idx]], device=device)

    return {
        "tokens": sampled_tokens,
        "entropies": entropies,
        "is_trigger": triggers,
        "H_first": H_first,
        "alpha_target": alpha_target,
        "trigger_positions": trigger_positions,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="HF model dir or id")
    p.add_argument("--prompts", required=True, help="parquet with 'prompt' column")
    p.add_argument("--n_prompts", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=2000)
    # Langevin hyperparameters (defaults match the paper §4)
    p.add_argument("--top_k", type=int, default=20, dest="K")
    p.add_argument("--langevin_eta", type=float, default=0.1)
    p.add_argument("--langevin_sigma", type=float, default=0.01)
    p.add_argument("--p_quantile", type=float, default=0.85)
    p.add_argument("--t_min", type=int, default=800)
    p.add_argument("--alpha_target", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.7, help="drift Polyak momentum")
    p.add_argument("--alpha", type=float, default=0.6, help="drift exploit ratio")
    # Misc
    p.add_argument("--tag", required=True)
    p.add_argument("--out_dir", default="./entropy_out")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / f"entropy__{args.tag}.jsonl"
    if out_path.exists():
        print(f"[have] {out_path} (delete to re-run)", file=sys.stderr)
        sys.exit(0)

    print(f"[entropy_ts] tag={args.tag}  ckpt={args.ckpt}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()

    ds = datasets.Dataset.from_parquet(args.prompts)
    ds = ds.select(range(min(args.n_prompts, len(ds))))

    with open(out_path, "w") as f:
        for i, ex in enumerate(ds):
            msgs = ex["prompt"]
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            else:
                prompt = msgs[0]["content"]

            for cond in ("vanilla", "drift"):
                use_langevin = cond == "drift"
                print(f"[entropy_ts] prompt {i}  cond={cond}", file=sys.stderr)
                result = rollout(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    K=args.K,
                    eta=args.langevin_eta,
                    sigma=args.langevin_sigma,
                    p_quantile=args.p_quantile,
                    t_min=args.t_min,
                    alpha_target=args.alpha_target,
                    gamma=args.gamma,
                    alpha=args.alpha,
                    use_langevin=use_langevin,
                    seed=args.seed + i,
                )
                rec = {
                    "tag": args.tag,
                    "prompt_idx": i,
                    "condition": cond,
                    **result,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()

    print(f"[entropy_ts] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
