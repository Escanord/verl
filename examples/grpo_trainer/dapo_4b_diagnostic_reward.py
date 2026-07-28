"""Diagnostic reward function for DAPO 4B eval.

Scores each rollout with BOTH extractor modes and logs both, plus a short
head/tail of the model output so we can see whether the model emits an
`Answer:` prefix, a `\boxed{}` box, or something else.

Called by verl's DAPO reward manager: (data_source, solution_str,
ground_truth, extra_info) → dict{score, acc, pred, ...}.
"""

import os
import re
import sys

from verl.utils.reward_score import math_dapo

_LOG_EVERY = int(os.environ.get("DAPO_DIAG_LOG_EVERY", "1"))
_HEAD_CHARS = int(os.environ.get("DAPO_DIAG_HEAD_CHARS", "0"))
_TAIL_CHARS = int(os.environ.get("DAPO_DIAG_TAIL_CHARS", "500"))
_COUNTER = {"n": 0}


def _has_answer_prefix(s: str) -> bool:
    return bool(re.search(r"(?i)Answer\s*:\s*[^\n]", s))


def _has_boxed(s: str) -> bool:
    return "\\boxed{" in s


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Ground truth may be a dict like {'ground_truth': '...', 'style': 'rule'}.
    if isinstance(ground_truth, dict):
        gt_str = ground_truth.get("ground_truth", "")
    else:
        gt_str = str(ground_truth)

    # Both extractor modes on the same rollout.
    result_minerva = math_dapo.compute_score(solution_str, gt_str, strict_box_verify=False)
    result_box    = math_dapo.compute_score(solution_str, gt_str, strict_box_verify=True)

    minerva_acc = 1 if result_minerva.get("acc") else 0
    box_acc     = 1 if result_box.get("acc") else 0

    _COUNTER["n"] += 1
    if _LOG_EVERY > 0 and _COUNTER["n"] % _LOG_EVERY == 0:
        head = solution_str[:_HEAD_CHARS] if _HEAD_CHARS > 0 else ""
        tail = solution_str[-_TAIL_CHARS:]
        sep = " ... " if _HEAD_CHARS > 0 else ""
        print(
            f"[DAPO-DIAG #{_COUNTER['n']:04d}] "
            f"gt={gt_str!r} "
            f"minerva_acc={minerva_acc} minerva_pred={result_minerva.get('pred')!r} "
            f"box_acc={box_acc} box_pred={result_box.get('pred')!r} "
            f"has_Answer={_has_answer_prefix(solution_str)} "
            f"has_boxed={_has_boxed(solution_str)} "
            f"resp_len={len(solution_str)}\n"
            f"    tail: {tail!r}",
            flush=True,
        )
        sys.stdout.flush()

    # Use box score as the returned reward — the paper's likely extractor.
    # (Doesn't matter for diagnostic; we log both above.)
    return {
        "score":       result_box["score"],
        "acc":         box_acc,
        "pred":        result_box.get("pred"),
        "minerva_acc": minerva_acc,
        "box_acc":     box_acc,
    }
