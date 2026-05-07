"""
Custom reward function for guru-RL-92k dataset and DRIFT eval suite.

Returns a dict per rollout containing the standard {"score", "acc", "pred"}
keys plus per-rollout diagnostics {"response_length", "entropy"} when those
are available in extra_info (populated by the naive reward manager).  All
keys flow through verl's existing reward_extra_info pipeline and are
aggregated by process_validation_metrics as
val-core/<data_source>/<key>/{mean@N, best@N/mean, ...}.

Data-source dispatch:
    math__*       -> math_dapo verifier (sympy-based answer-equivalence checking)
                     covers AIME24/25, MATH-500, OlympiadBench-Math-EN, Minerva-Math.
    mcq__*        -> simple MCQ regex match on 'Answer: $LETTER' (A-D);
                     covers GPQA-Diamond.
"""

import re

from verl.utils.reward_score import math_dapo

_MCQ_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")


def _score_mcq(solution_str: str, ground_truth: str) -> dict:
    m = _MCQ_PATTERN.search(solution_str or "")
    pred = m.group(1).upper() if m else None
    correct = pred is not None and pred == str(ground_truth).upper()
    return {
        "score": 1.0 if correct else -1.0,
        "acc": bool(correct),
        "pred": pred if pred is not None else "",
    }


def _attach_diagnostics(out: dict, extra_info) -> dict:
    """Bubble response_length / entropy from extra_info into the reward dict
    so they flow through verl's reward_extra_info pipeline as per-rollout
    diagnostics.  No-ops if extra_info is missing those keys (e.g. when
    invoked outside the patched naive reward manager)."""
    if not isinstance(extra_info, dict):
        return out
    rl = extra_info.get("response_length")
    if rl is not None:
        out["response_length"] = float(rl)
    ent = extra_info.get("entropy")
    if ent is not None:
        out["entropy"] = float(ent)
    return out


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source.startswith("math__"):
        result = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source.startswith("mcq__"):
        result = _score_mcq(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"No reward function for {data_source=}")
    return _attach_diagnostics(result, extra_info)
