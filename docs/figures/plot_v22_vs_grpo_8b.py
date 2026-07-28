#!/usr/bin/env python3
"""Regenerate docs/figures/v22_vs_grpo_8b_step{MAX}.png.

2x4 grid: rows = {mean@16, best@16}, cols = {AIME24, AIME25, MATH-500,
OlympBench}; two lines per panel (GRPO vs DRIFT-v22) over training step.
Reads the verl file-logger jsonls directly (metrics live under the "data"
key, val accuracies as val-core/<bench>/acc/{mean@16, best@16/mean}).

Run with an env that has matplotlib, e.g.:
  /home/escanord/duy/venv-vault/miniconda3/envs/fafo/bin/python \
      docs/figures/plot_v22_vs_grpo_8b.py 400
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 400

# name -> (color, training-jsonl, backfill-eval tag).  AIME25 / OlympBench were
# not in the 8B training-time val set (GRPO has them nowhere, v22 only from
# step 205), so those early points come from the standalone backfill eval
# passes under eval_out/<tag>_backfill/<tag>_step<N>/.
RUNS = {
    "GRPO": ("#4C72B0",
             "/home/escanord/duy/checkpoints/verl/grpo/verl_grpo_qwen3_guru_rl/qwen3_8b_base_grpo.jsonl",
             "grpo_8b"),
    "DRIFT-v22": ("#C44E52",
                  "/home/escanord/duy/checkpoints/verl/pivot-v22/verl_pivot_v22_qwen3_guru_rl/qwen3_8b_base_pivot_v22.jsonl",
                  "v22_8b"),
}
BACKFILL_ROOT = "/home/escanord/duy/checkpoints/verl/eval_out"
# (column title, data_source key)
BENCH = [
    ("AIME24", "math__aime_repeated_8x"),
    ("AIME25", "math__aime25"),
    ("MATH-500", "math__math"),
    ("OlympBench", "math__olympiadbench_en"),
]
METRICS = [("mean@16", "mean@16"), ("best@16", "best@16/mean")]


def _from_training(path, key):
    """{step: value} for a val-core key, merging every resumed/appended chunk.

    The verl file logger appends every resumed segment and every backfill
    eval pass to the same jsonl, so a single training trajectory is split
    across several out-of-order chunks (e.g. steps 5..335, then 485..795).
    Merge ALL val rows and dedupe by step rather than slicing to the last
    chunk — which would drop most of the curve.
    """
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        d = rec.get("data")
        if isinstance(d, dict) and key in d and d[key] is not None:
            out[rec["step"]] = d[key]
    return out


def _from_backfill(tag, key):
    """{step: value} from standalone backfill eval passes, keyed by ckpt step."""
    out = {}
    for d in glob.glob(f"{BACKFILL_ROOT}/{tag}_backfill/{tag}_step*"):
        base = os.path.basename(d)
        try:
            step = int(base.split("_step")[-1])
        except ValueError:
            continue
        js = glob.glob(f"{d}/**/*.jsonl", recursive=True)
        if not js:
            continue
        for line in open(js[0]):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dd = rec.get("data", rec)
            if isinstance(dd, dict) and key in dd and dd[key] is not None:
                out[step] = dd[key]
    return out


def series(path, tag, bench, metric_suffix):
    """(steps, acc%) merging backfill eval + training log; training wins on tie."""
    key = f"val-core/{bench}/acc/{metric_suffix}"
    merged = _from_backfill(tag, key)
    merged.update(_from_training(path, key))  # native training log takes precedence
    xs = sorted(s for s in merged if s <= MAX_STEP)
    return xs, [merged[s] * 100.0 for s in xs]


fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True)
for r, (mlabel, msuffix) in enumerate(METRICS):
    for c, (btitle, bkey) in enumerate(BENCH):
        ax = axes[r][c]
        for name, (color, path, tag) in RUNS.items():
            xs, ys = series(path, tag, bkey, msuffix)
            if not xs:
                continue
            ax.plot(xs, ys, color=color, lw=1.7, label=name)
            # peak star + annotation
            pi = max(range(len(ys)), key=lambda i: ys[i])
            ax.plot(xs[pi], ys[pi], marker="*", ms=11, color=color,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=5)
            ax.annotate(f"{ys[pi]:.1f}", (xs[pi], ys[pi]),
                        textcoords="offset points", xytext=(2, 6),
                        color=color, fontsize=9, fontweight="bold")
        ax.set_title(f"{btitle} · {mlabel}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, MAX_STEP)
        if c == 0:
            ax.set_ylabel("acc (%)")
        if r == 1:
            ax.set_xlabel("training step")

handles = [plt.Line2D([], [], color=col, lw=2.2, label=n) for n, (col, _, _) in RUNS.items()]
fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955),
           frameon=False, fontsize=12)
fig.suptitle(f"DRIFT-v22 vs GRPO on Qwen3-8B-Base — training steps 0–{MAX_STEP}",
             fontsize=16, fontweight="bold", y=0.99)
footer = (f"val: n=16, T=1.0, top-p=1.0    ★ = peak within [0, {MAX_STEP}]\n"
          "GRPO: train_bsz=1024, KL_loss=0.001, n=8\n"
          "DRIFT-v22: KL_in_reward β=2.0 + Langevin + soft-IS + peak-ratchet t_min")
fig.text(0.008, 0.005, footer, fontsize=9, color="#444444", va="bottom", family="monospace")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
out = f"/home/escanord/duy/verl/docs/figures/v22_vs_grpo_8b_step{MAX_STEP}.png"
fig.savefig(out, dpi=130)
print("saved", out)
