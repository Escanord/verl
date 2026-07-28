#!/usr/bin/env python3
"""Entropy -> reward-trajectory reachability probe (rebuttal artifact B).

Reviewer question: "why should preserving downstream entropy correlate with
discovering higher-reward trajectories?"

Mechanistic answer this probe tests: a policy can only *reinforce* trajectories
it can *sample*.  A correct-but-not-yet-mastered trajectory is reachable only
if the policy keeps non-negligible probability on it — i.e. only if it retains
entropy.  Collapse the entropy and the correct trajectory drops out of the
sampling support -> pass@k -> 0 -> RL can never discover it.

We use sampling temperature as a *clean instrument* for the policy's output
entropy on a FIXED checkpoint (isolates "spread of the distribution" from every
other training variable), sweep it, and measure:
    - realized mean per-token entropy (from top-k logprobs)
    - coverage = pass@k = frac of prompts solved at least once in k samples
      (= reachability of a correct trajectory)
    - mean@k = average per-sample accuracy
The prediction: coverage collapses toward 0 as entropy -> 0.  That is the
causal statement "entropy governs reward-trajectory reachability", which is the
missing link behind "preserve entropy -> discover higher reward".

Run (needs 1 GPU):
    python examples/empirical/entropy_reachability_probe.py \
        --model /home/escanord/duy/checkpoints/models/Qwen3-4B-Base \
        --data  /home/escanord/duy/data/guru_rl/test_math500.parquet \
        --n_prompts 200 --k 16 --temps 0.2,0.5,0.8,1.0,1.3 \
        --out /home/escanord/duy/checkpoints/verl/empirical/reachability_qwen3_4b_math500.json
"""
import argparse
import json
import math
import os

import pandas as pd

from verl.utils.reward_score import math_dapo


def per_token_entropy(logprobs_seq):
    """Truncated per-token entropy from vLLM top-k logprobs (nats), averaged
    over the generated tokens.  Truncated to the returned top-k, so it is a
    lower bound on the true entropy but monotone in temperature — fine for the
    trend."""
    ents = []
    for step_lp in logprobs_seq or []:
        if not step_lp:
            continue
        lps = [lp.logprob for lp in step_lp.values()]
        h = -sum(math.exp(x) * x for x in lps)
        ents.append(h)
    return sum(ents) / len(ents) if ents else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temps", default="0.2,0.5,0.8,1.0,1.3")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    temps = [float(t) for t in args.temps.split(",")]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    df = pd.read_parquet(args.data).iloc[: args.n_prompts]

    def render(msgs):
        try:
            return tok.apply_chat_template(list(msgs), tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(m["content"] for m in msgs)

    prompts = [render(r["prompt"]) for _, r in df.iterrows()]
    gts = [r["reward_model"]["ground_truth"] for _, r in df.iterrows()]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_tokens + 2048, trust_remote_code=True)

    results = []
    for T in temps:
        sp = SamplingParams(n=args.k, temperature=T, top_p=1.0, max_tokens=args.max_tokens,
                            logprobs=20)
        outs = llm.generate(prompts, sp)
        n_solved = 0            # prompts solved at least once (coverage / pass@k)
        per_sample_acc = []     # every (prompt,sample) correctness -> mean@k
        ent_acc = []
        for out, gt in zip(outs, gts):
            solved = False
            for comp in out.outputs:
                sc = math_dapo.compute_score(comp.text, gt, strict_box_verify=True)
                correct = bool(sc.get("acc"))
                per_sample_acc.append(1.0 if correct else 0.0)
                solved = solved or correct
                ent_acc.append(per_token_entropy(comp.logprobs))
            if solved:
                n_solved += 1
        cov = n_solved / len(prompts)
        mean_k = sum(per_sample_acc) / len(per_sample_acc)
        ent = sum(e for e in ent_acc if e == e) / max(1, sum(1 for e in ent_acc if e == e))
        row = {"temperature": T, "realized_entropy": ent, "coverage_passk": cov, "mean_k": mean_k,
               "n_prompts": len(prompts), "k": args.k}
        results.append(row)
        print(f"T={T:>4}  entropy={ent:6.3f}  coverage(pass@{args.k})={cov*100:5.1f}%  mean@{args.k}={mean_k*100:5.1f}%",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "data": args.data, "results": results}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
