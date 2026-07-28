"""
Custom reward function for guru-RL-92k dataset and DRIFT eval suite.

Returns a dict per rollout containing the standard {"score", "acc", "pred"}
keys plus per-rollout diagnostics {"response_length", "entropy"} when those
are available in extra_info (populated by the naive reward manager).  All
keys flow through verl's existing reward_extra_info pipeline and are
aggregated by process_validation_metrics as
val-core/<data_source>/<key>/{mean@N, best@N/mean, ...}.

Data-source dispatch:
    math__*       -> math_dapo verifier with strict_box_verify=True
                     (extracts from the last \\boxed{...}).  This is the
                     single canonical answer format used across train and
                     val: our prompts consistently ask the model to place
                     the final answer within \\boxed{}, matching the 80/20
                     paper's chat template convention.
                     (Historical note: earlier versions of this file left
                     math_dapo.compute_score at its default strict_box_verify=False
                     which requires an "Answer:" prefix regex.  Because our
                     prompts didn't ask for that format, ~70% of the model's
                     correct rollouts were being scored as wrong at both
                     train and val time — a silent misscoring that
                     under-reported DAPO val by ~15× and mis-directed the
                     DAPO training gradient.  Fixed to strict_box_verify=True
                     to align train/val/paper reference.)
                     Covers AIME24/25, MATH-500, OlympiadBench-Math-EN,
                     Minerva-Math.
    mcq__*        -> simple MCQ regex match on 'Answer: $LETTER' (A-D);
                     covers GPQA-Diamond.

Reported per-rollout keys used by the paper:
    score        : reward in {+1, -1}
    acc          : same as score converted to bool — feeds val-core/*/acc/mean@16
    pred         : normalized prediction extracted from \\boxed{...}
"""

import re

from verl.utils.reward_score import math_dapo

# Primary format requested by the GPQA prompt: "Answer: $LETTER" (also
# tolerate "Answer: (C)" / "Answer: $C$").
_MCQ_ANSWER_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\(?([A-D])\)?\$?")
# Fallback: a model RL-trained on math (where \boxed{} is the rewarded format)
# frequently emits the MCQ letter inside \boxed{...} instead of "Answer: X".
# Without this fallback those genuinely-correct answers would be scored wrong,
# under-counting GPQA and biasing it by format-compliance drift over training.
_MCQ_BOXED_PATTERN = re.compile(r"(?i)\\boxed\{\s*\(?([A-D])\)?\s*\}")


def _extract_mcq_letter(solution_str: str):
    """Extract the chosen MCQ letter (A-D) from either the prompt-requested
    'Answer: X' format or a \\boxed{X} fallback.  Prefer an explicit 'Answer:'
    (the instructed format); else fall back to a boxed letter.  Take the LAST
    match so a final answer overrides earlier tentative ones."""
    s = solution_str or ""
    matches = _MCQ_ANSWER_PATTERN.findall(s) or _MCQ_BOXED_PATTERN.findall(s)
    return matches[-1].upper() if matches else None


def _score_mcq(solution_str: str, ground_truth: str) -> dict:
    pred = _extract_mcq_letter(solution_str)
    correct = pred is not None and pred == str(ground_truth).upper()
    return {
        "score": 1.0 if correct else -1.0,
        "acc": bool(correct),
        "pred": pred if pred is not None else "",
    }


def _attach_diagnostics(out: dict, extra_info) -> dict:
    """Bubble response_length / entropy from extra_info into the reward dict
    so they flow through verl's reward_extra_info pipeline as per-rollout
    diagnostics.

    IMPORTANT: verl's naive reward manager only appends keys that appear in
    the reward function's return dict (workers/reward_manager/naive.py:105).
    If different rollouts return different key sets, `reward_extra_info`
    ends up with mismatched list lengths, which breaks the downstream
    `var_vals[sample_idx]` indexing in `process_validation_metrics`.
    Attach both keys unconditionally with 0.0 defaults so the schema is
    identical across every rollout of every data source that goes through
    this reward function."""
    ei = extra_info if isinstance(extra_info, dict) else {}
    rl = ei.get("response_length")
    out["response_length"] = float(rl) if rl is not None else 0.0
    ent = ei.get("entropy")
    out["entropy"] = float(ent) if ent is not None else 0.0
    return out


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source.startswith("math__"):
        result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    elif data_source.startswith("mcq__"):
        result = _score_mcq(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"No reward function for {data_source=}")
    # verl's process_validation_metrics uses `isinstance(var_vals[0], str)` to
    # skip string-valued fields when computing mean@N.  If pred is None on
    # the first sample but str on later ones, np.mean() sees a mixed
    # None+str array and crashes.  Coerce pred to always be a string.
    if result.get("pred") is None:
        result["pred"] = ""
    return _attach_diagnostics(result, extra_info)
