"""
Custom reward function for guru-RL-92k code subset (LeetCode/PrimeIntellect/
TACO for train; HumanEval/MBPP/LiveCodeBench for eval).

Returns a dict per rollout containing {"score", "acc", "pred"} plus optional
diagnostics.  Score is binary: +1.0 if all tests pass, -1.0 otherwise —
preserving the dead-zone story from the math setup (GRPO advantage is 0 for
all-correct / all-wrong groups).

Two test formats supported:
    stdin/stdout  ({"inputs": [...], "outputs": [...]})
        codegen__primeintellect, codegen__taco
        → spawn a subprocess `python -c <user_code>`, pipe each input as
        stdin, compare captured stdout to expected output (normalized).
    functional    ({"functional": "def check(candidate): assert ..."})
        codegen__leetcode2k, codegen__humaneval  (check(candidate) style)
        codegen__mbpp  (raw asserts style)
        → subprocess execs the user code, then runs the test harness,
        catches AssertionError / exceptions.

All execution is sandboxed via a fresh Python subprocess with a hard timeout.
"""

import json
import re
import subprocess
import sys
import textwrap

_PY_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_STDIN_STDOUT_TIMEOUT = 8       # per test case
_FUNCTIONAL_TIMEOUT   = 15      # for the full harness


def _extract_code(solution_str: str) -> str:
    """Grab the last ```python``` block; fall back to whole string."""
    matches = _PY_BLOCK.findall(solution_str or "")
    if matches:
        return matches[-1].strip()
    return (solution_str or "").strip()


def _run_python(script: str, stdin: str = "", timeout: int = 8) -> tuple[int, str, str]:
    """Run `script` in a fresh Python subprocess with `stdin`; return (rc, stdout, stderr)."""
    try:
        p = subprocess.run(
            [sys.executable, "-c", script],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", f"{type(e).__name__}: {e}"


def _norm(s: str) -> str:
    """Normalize output for comparison: trim, collapse trailing whitespace per line."""
    if s is None:
        return ""
    return "\n".join(line.rstrip() for line in s.rstrip("\n").split("\n"))


def _score_stdin_stdout(solution_str: str, gt: dict, max_tests: int = 50) -> dict:
    """Run user's code against a batch of (input, expected_output) pairs."""
    user_code = _extract_code(solution_str)
    if not user_code:
        return {"score": -1.0, "acc": False, "pred": "no_code_block"}
    inputs  = gt.get("inputs", [])
    outputs = gt.get("outputs", [])
    if len(inputs) != len(outputs) or not inputs:
        return {"score": -1.0, "acc": False, "pred": "malformed_tests"}

    # Cap test count to keep step time bounded.  Raised 15 -> 50 to cut the
    # false-positive rate (a solution passing only the first K tests was being
    # scored fully correct).  Note: one subprocess is spawned PER test here, so
    # very high caps get expensive once the policy is strong; the real fix is to
    # run all tests inside a single subprocess.
    n = min(len(inputs), max_tests)
    fails = 0
    for i in range(n):
        rc, out, err = _run_python(user_code, stdin=inputs[i], timeout=_STDIN_STDOUT_TIMEOUT)
        if rc != 0 or _norm(out) != _norm(outputs[i]):
            fails += 1
            break
    correct = (fails == 0)
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": f"passed_{n-fails}/{n}"}


def _score_functional(solution_str: str, harness: str, mode: str) -> dict:
    """Execute user's code + a test harness in a fresh subprocess."""
    user_code = _extract_code(solution_str)
    if not user_code:
        return {"score": -1.0, "acc": False, "pred": "no_code_block"}

    if mode == "check_candidate":
        # Runner script:
        #   1. exec user code, then exec harness (which defines `check(candidate)`)
        #   2. discover the candidate object:
        #        a. `candidate = Solution().<method>` if the harness signature says so
        #        b. `check(<name>)` if the harness explicitly calls check with a name
        #        c. Solution().<first_public_method>  (LeetCode fallback)
        #        d. any user-defined callable in _env whose name isn't "check"
        #   3. call check(candidate); success = no AssertionError / no exception
        runner = textwrap.dedent("""
        import sys, re, io, contextlib, inspect
        _stdout = io.StringIO(); _stderr = io.StringIO()
        _BUILTIN_KEYS = None  # snapshot before user code runs
        try:
            with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
                _env = {"__name__": "__main__"}
                from functools import cache, lru_cache
                from typing import *
                import math, itertools, collections, heapq, bisect, re as _re
                _env.update({k: v for k, v in list(locals().items())
                             if k not in ('_env','_stdout','_stderr','sys','re','io','contextlib','inspect','_BUILTIN_KEYS')})
                _BUILTIN_KEYS = set(_env.keys())
                exec(_USER_CODE, _env)
                _USER_KEYS = [k for k in _env.keys() if k not in _BUILTIN_KEYS]  # definition order -> deterministic candidate pick
                exec(_HARNESS, _env)
                check = _env.get("check")
                if check is None:
                    raise RuntimeError("no check() in harness")
                candidate = None
                # (a) explicit `candidate = Solution().<method>` in harness
                m = _re.search(r"candidate\\s*=\\s*Solution\\(\\)\\.(\\w+)", _HARNESS)
                if m and "Solution" in _env:
                    inst = _env["Solution"]()
                    candidate = getattr(inst, m.group(1), None)
                # (b) `check(<name>)` explicitly invoked in the harness with a function name
                if candidate is None:
                    m = _re.search(r"check\\s*\\(\\s*(\\w+)\\s*\\)", _HARNESS)
                    if m and m.group(1) in _env and callable(_env[m.group(1)]):
                        candidate = _env[m.group(1)]
                # (c) any Solution method (LeetCode style, no explicit method in harness)
                if candidate is None and "Solution" in _env:
                    inst = _env["Solution"]()
                    methods = [n for n in dir(inst) if not n.startswith("_") and callable(getattr(inst, n))]
                    candidate = getattr(inst, methods[0]) if methods else None
                # (d) any user-defined callable that isn't the check function
                if candidate is None:
                    for k in _USER_KEYS:
                        if k == "check":
                            continue
                        v = _env.get(k)
                        if callable(v):
                            candidate = v
                            break
                if candidate is None:
                    raise RuntimeError("no candidate found (user keys: " + str(sorted(_USER_KEYS)) + ")")
                check(candidate)
            print("__PASS__")
        except AssertionError as e:
            print("__FAIL__:AssertionError:" + str(e))
        except Exception as e:
            print("__ERROR__:" + type(e).__name__ + ":" + str(e))
        """)
    elif mode == "raw_asserts":
        runner = textwrap.dedent("""
        import io, contextlib
        _stdout = io.StringIO(); _stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
                _env = {"__name__": "__main__"}
                from functools import cache, lru_cache
                from typing import *
                import math, itertools, collections, heapq, bisect, re
                _env.update({k: v for k, v in list(locals().items())
                             if k not in ('_env','_stdout','_stderr','io','contextlib')})
                exec(_USER_CODE, _env)
                exec(_HARNESS, _env)
            print("__PASS__")
        except AssertionError as e:
            print("__FAIL__:AssertionError:" + str(e))
        except Exception as e:
            print("__ERROR__:" + type(e).__name__ + ":" + str(e))
        """)
    else:
        return {"score": -1.0, "acc": False, "pred": f"unknown_mode:{mode}"}

    # Bake user_code and harness into the runner string as literals to avoid escaping issues:
    # write them to sys.argv accessible constants via a preamble.
    preamble = f"_USER_CODE = {user_code!r}\n_HARNESS = {harness!r}\n"
    rc, out, err = _run_python(preamble + runner, stdin="", timeout=_FUNCTIONAL_TIMEOUT)
    if "__PASS__" in out:
        return {"score": 1.0, "acc": True, "pred": "pass"}
    if "__FAIL__" in out:
        # Extract just the fail reason
        line = [l for l in out.splitlines() if l.startswith("__FAIL__")][-1]
        return {"score": -1.0, "acc": False, "pred": line[:120]}
    if "__ERROR__" in out:
        line = [l for l in out.splitlines() if l.startswith("__ERROR__")][-1]
        return {"score": -1.0, "acc": False, "pred": line[:120]}
    # Runner didn't finish (timeout / crash)
    return {"score": -1.0, "acc": False, "pred": f"rc={rc}_err={err[:80] if err else ''}"}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if isinstance(ground_truth, dict):
        gt = ground_truth
    else:
        try:
            gt = json.loads(ground_truth)
        except Exception:
            gt = {"raw": ground_truth}
    if not isinstance(gt, dict):
        gt = {"raw": gt}

    if not data_source.startswith("codegen__"):
        raise NotImplementedError(f"No reward function for {data_source=}")

    if data_source == "codegen__mbpp":
        return _score_functional(solution_str, gt.get("functional", ""), mode="raw_asserts")
    if "functional" in gt:
        return _score_functional(solution_str, gt["functional"], mode="check_candidate")
    if "inputs" in gt and "outputs" in gt:
        return _score_stdin_stdout(solution_str, gt)
    return {"score": -1.0, "acc": False, "pred": f"unrecognized_gt_keys:{list(gt.keys())}"}
