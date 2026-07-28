#!/usr/bin/env python3
"""Fused dead-zone diagnostic at a checkpoint (rebuttal artifact, 4B cold start).

Reviewer: "why should preserving downstream entropy correlate with discovering
higher-reward trajectories?"  We answer it at the branch-point level with TWO
quantities measured on the SAME rollouts, at 4B step 40, GRPO vs DRIFT:

  y  = mixed-correctness %  (within-group reward variance = the GRPO signal:
       fraction of n-rollout groups with BOTH a correct and an incorrect rollout)

  x  = Delta = mean( H_after(W) - H_branch )  over branch points
       ("entropy after the branch point"): how much per-token entropy DROPS in
       the W tokens following each high-entropy fork.  GRPO is expected to be
       strongly negative (forks once, then commits -> entropy collapses); DRIFT
       ~0 (maintains downstream branching).  Non-circular because the window
       averages ordinary tokens, not the fork itself.

Config follows the ACTUAL DRIFT-4B run (v22_4b_*.err), not the paper text:
  AIME-24 prompts, n=8, temp 1.0, max_resp 4096; branch point = per-token
  entropy above the p=0.85 quantile of the pooled response-token entropies,
  past position 200; entropy is the UNPERTURBED pi_theta entropy (Eq. 2's
  trigger), computed by an exact HF forward pass over the vLLM-generated
  rollouts (the paper's Table-7 method).

Scoring uses the training verifier (math_dapo, strict_box_verify=True).
"""
import argparse
import json
import os

import pandas as pd
import torch

from verl.utils.reward_score import math_dapo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF checkpoint dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_prompts", type=int, default=30)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--branch_quantile", type=float, default=0.85)  # p=0.85 trig_percentile
    ap.add_argument("--pos_guard", type=int, default=200)           # peak-mode floor
    ap.add_argument("--window", type=int, default=32)               # W for H_after
    ap.add_argument("--forkable_bar", type=float, default=0.4)      # tau0, for the 2nd lens
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    df = pd.read_parquet(args.data)
    df["_key"] = df["prompt"].apply(lambda m: "".join(x["content"] for x in m))
    df = df.drop_duplicates("_key").iloc[: args.n_prompts].reset_index(drop=True)

    def render(msgs):
        try:
            return tok.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(m["content"] for m in msgs)

    prompts = [render(r["prompt"]) for _, r in df.iterrows()]
    gts = [r["reward_model"]["ground_truth"] for _, r in df.iterrows()]

    # ---- 1. generate rollouts (vLLM) ----------------------------------------
    from vllm import LLM, SamplingParams
    # enforce_eager=True: skip vLLM's torch.compile/inductor path, which shells
    # out to `nvcc --version` — nvcc isn't installed on these nodes (verl's
    # training rollout runs eager for the same reason).
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_tokens + 2048, trust_remote_code=True,
              seed=args.seed, enforce_eager=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)
    # collect (prompt_idx, response_token_ids, text)
    rollouts = []
    for pi, out in enumerate(outs):
        for comp in out.outputs:
            rollouts.append((pi, list(comp.token_ids), comp.text))
    del llm
    torch.cuda.empty_cache()

    # ---- 2. reward variance (mixed-correctness) -----------------------------
    by_prompt = {}
    for pi, _, text in rollouts:
        sc = math_dapo.compute_score(text, gts[pi], strict_box_verify=True)
        by_prompt.setdefault(pi, []).append(bool(sc.get("acc")))
    ng = len(by_prompt)
    mixed = sum(1 for v in by_prompt.values() if 0 < sum(v) < len(v))
    total_correct = sum(sum(v) for v in by_prompt.values())
    total = sum(len(v) for v in by_prompt.values())

    # ---- 3. exact per-token entropy (HF forward) ----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    all_H = []            # per-token response entropies (for the quantile)
    seq_H = []            # list of (prompt_idx, [H per response token])
    with torch.no_grad():
        for pi, resp_ids, _ in rollouts:
            if not resp_ids:
                continue
            p_ids = tok(prompts[pi], add_special_tokens=False)["input_ids"]
            full = torch.tensor([p_ids + resp_ids], device="cuda")
            logits = model(full).logits[0]                      # (T, V)
            # entropy of the distribution that PRODUCED each response token:
            start = len(p_ids) - 1
            end = start + len(resp_ids)                          # len(resp) positions
            lg = logits[start:end].float()
            logp = torch.log_softmax(lg, dim=-1)
            H = -(logp.exp() * logp).sum(-1)                    # (len(resp),) nats
            Hl = H.tolist()
            seq_H.append((pi, Hl))
            all_H.extend(Hl)

    thr = float(pd.Series(all_H).quantile(args.branch_quantile))

    # ---- 4. Delta = H_after(W) - H_branch over branch points ----------------
    W = args.window
    deltas = []
    branch_H = []
    n_forkable = n_pos = 0
    for pi, Hl in seq_H:
        L = len(Hl)
        for t in range(L):
            if t < args.pos_guard:
                continue
            n_pos += 1
            if Hl[t] > args.forkable_bar:
                n_forkable += 1
            if Hl[t] > thr and t + 1 < L:  # branch point with room for a window
                after = Hl[t + 1: min(L, t + 1 + W)]
                if after:
                    deltas.append(sum(after) / len(after) - Hl[t])
                    branch_H.append(Hl[t])

    import statistics as st
    summary = {
        "label": args.label,
        "model": args.model,
        "n_prompts": ng,
        "n": args.n,
        # y-axis
        "mixed_correctness_pct": 100.0 * mixed / ng,
        "solve_pct": 100.0 * total_correct / total,
        # x-axis (entropy after the branch)
        "branch_quantile": args.branch_quantile,
        "branch_entropy_threshold_nats": thr,
        "n_branch_points": len(deltas),
        "mean_H_branch_nats": st.mean(branch_H) if branch_H else float("nan"),
        "mean_delta_after_branch_nats": st.mean(deltas) if deltas else float("nan"),
        "median_delta_after_branch_nats": st.median(deltas) if deltas else float("nan"),
        # 2nd lens: absolute branching capacity
        "frac_positions_above_%.1fnats" % args.forkable_bar: 100.0 * n_forkable / max(1, n_pos),
        "window_W": W,
        "pos_guard": args.pos_guard,
    }
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
