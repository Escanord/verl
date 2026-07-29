#!/usr/bin/env python3
"""Numeric-equivalence harness for the DRIFT Langevin processor vectorization.

Goal: prove a *batched* implementation of the per-token/per-trigger math produces
identical results to the active v2 per-request logic, so the vectorized rollout
processor does not silently change the method.

We compare, on identical (logits, G, prev_eps, H_first, has_trig, step, noise):
  1. trigger mask   = (H > entropy_threshold) & (step >= effective_tmin)
  2. perturbation eps  (v2 _langevin_step_feedback per row  vs  batched)
  3. G momentum update (v2 formula per row  vs  batched, Bug-A-fixed)
  4. IS top-k log_p_lan of the perturbed logits
  5. H (entropy)

Run: python examples/empirical/test_pivot_vectorization.py
"""
import torch

from verl.utils.vllm import pivot_patch as pp


def entropy(logits):  # (…, V) -> (…)
    p = torch.softmax(logits.float(), dim=-1)
    return -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1)


# ----- config mirrors the real DRIFT-4B run -----------------------------------
SIGMA = 0.01
TOP_K = 128
EXPLOIT = 0.9          # langevin_exploit_ratio (alpha in the mixing)
ALPHA_TARGET = 0.7     # langevin_alpha_target
GAMMA = 0.7            # langevin_momentum
TAU = 0.4              # entropy_threshold
TMIN = 400             # effective_tmin (static)
EOS_IDS = torch.tensor([1, 2], dtype=torch.long)


def reference_per_row(logits, G, prev_eps, H_first, has_trig, step, noise):
    """v2 semantics, computed one row at a time with the existing helpers."""
    N, V = logits.shape
    H = entropy(logits)                                   # (N,)
    trig = (H > TAU) & (step >= TMIN)                     # (N,) bool
    eps = torch.zeros_like(logits)
    new_logits = logits.clone()
    is_ids, is_vals = [], []
    new_G = G.clone()
    new_H_first = H_first.clone()
    for i in range(N):
        if not bool(trig[i]):
            is_ids.append(None); is_vals.append(None)
            continue
        Gi = G[i] if float(G[i].abs().sum()) > 1e-6 else None
        nl_i, eps_i = pp._langevin_step_feedback(
            logits[i], SIGMA, TOP_K, Gi, EXPLOIT,
            eos_ids_tensor=EOS_IDS, noise=noise[i],
        )
        new_logits[i] = nl_i
        eps[i] = eps_i
        # G-update: v2 only updates on a trigger that has a PRIOR trigger
        # (first trigger just records H_first, no G change).
        if bool(has_trig[i]):
            signal = H[i] - H_first[i] * ALPHA_TARGET
            feedback = signal * prev_eps[i]
            new_G[i] = GAMMA * G[i] + (1.0 - GAMMA) * feedback
        else:
            new_H_first[i] = H[i]
        # IS top-k of perturbed logits
        lp = torch.log_softmax(nl_i.float(), dim=-1)
        tv, ti = torch.topk(lp, TOP_K)
        is_ids.append(ti); is_vals.append(tv)
    return dict(H=H, trig=trig, eps=eps, new_logits=new_logits,
                new_G=new_G, new_H_first=new_H_first, is_ids=is_ids, is_vals=is_vals)


def vectorized(logits, G, prev_eps, H_first, has_trig, step, noise):
    """Candidate batched implementation (what will go into the adapter)."""
    N, V = logits.shape
    H = entropy(logits)
    trig = (H > TAU) & (step >= TMIN)
    trig_f = trig.float().unsqueeze(-1)

    # perturbation for ALL rows (masked to triggers at the end) — same math as
    # _langevin_step_feedback_batched, WITH the eos mask (Bug B fix).
    new_logits_all, eps_all = pp._langevin_step_feedback_batched(
        logits, SIGMA, TOP_K, G, EXPLOIT, eos_ids_tensor=EOS_IDS, noise=noise,
    )
    eps = eps_all * trig_f
    new_logits = torch.where(trig.unsqueeze(-1), new_logits_all, logits)

    # G-update — ONLY triggered rows that have triggered before (Bug A fix).
    do_update = trig & has_trig                            # (N,)
    first_time = trig & (~has_trig)
    new_H_first = torch.where(first_time, H, H_first)
    signal = (H - new_H_first * ALPHA_TARGET)              # (N,)
    feedback = signal.unsqueeze(-1) * prev_eps             # (N,V)
    G_upd = GAMMA * G + (1.0 - GAMMA) * feedback
    new_G = torch.where(do_update.unsqueeze(-1), G_upd, G)

    # IS top-k of perturbed logits (batched)
    lp = torch.log_softmax(new_logits.float(), dim=-1)
    is_vals_all, is_ids_all = torch.topk(lp, TOP_K, dim=-1)  # (N, top_k)
    return dict(H=H, trig=trig, eps=eps, new_logits=new_logits,
                new_G=new_G, new_H_first=new_H_first,
                is_ids_all=is_ids_all, is_vals_all=is_vals_all)


def main():
    torch.manual_seed(0)
    N, V = 8, 4000
    # mixed distributions: some peaked (low H, no trigger), some flat (high H)
    logits = torch.randn(N, V) * torch.tensor([[6.0], [6.0], [0.5], [0.5],
                                               [0.5], [0.5], [6.0], [0.5]])
    # a couple of slots without a G signal yet
    G = torch.randn(N, V) * 0.02
    G[0].zero_(); G[3].zero_()
    prev_eps = torch.randn(N, V) * 0.1
    H_first = torch.rand(N) * 3.0
    has_trig = torch.tensor([True, False, True, False, True, True, False, True])
    step = torch.tensor([500, 500, 500, 500, 300, 500, 500, 600])  # slot 4 below tmin
    noise = torch.randn(N, V)

    ref = reference_per_row(logits, G, prev_eps, H_first, has_trig, step, noise)
    vec = vectorized(logits, G, prev_eps, H_first, has_trig, step, noise)

    def chk(name, a, b, tol=1e-4):
        d = (a - b).abs().max().item() if a.numel() else 0.0
        ok = d <= tol
        print(f"  {'OK ' if ok else 'FAIL'} {name:22s} max|Δ|={d:.2e}")
        return ok

    allok = True
    allok &= chk("H", ref["H"], vec["H"])
    allok &= (ref["trig"] == vec["trig"]).all().item()
    print(f"  {'OK ' if (ref['trig']==vec['trig']).all() else 'FAIL'} trigger_mask           {ref['trig'].tolist()}")
    allok &= chk("eps (triggered)", ref["eps"], vec["eps"])
    # new_logits has -inf at masked positions; (-inf)-(-inf)=NaN, so compare the
    # -inf pattern and the finite entries separately.
    ra, rb = ref["new_logits"], vec["new_logits"]
    inf_match = ((ra == float("-inf")) == (rb == float("-inf"))).all().item()
    fin = torch.isfinite(ra) & torch.isfinite(rb)
    fin_d = (ra[fin] - rb[fin]).abs().max().item() if fin.any() else 0.0
    nl_ok = inf_match and fin_d <= 1e-4
    print(f"  {'OK ' if nl_ok else 'FAIL'} new_logits             inf_pattern={inf_match} finite max|Δ|={fin_d:.2e}")
    allok &= nl_ok
    allok &= chk("new_G", ref["new_G"], vec["new_G"])
    allok &= chk("new_H_first", ref["new_H_first"], vec["new_H_first"])
    # IS top-k: compare only triggered rows (ref stores per-row)
    is_ok = True
    for i in range(N):
        if ref["is_ids"][i] is None:
            continue
        di = (ref["is_ids"][i] - vec["is_ids_all"][i]).abs().max().item()
        dv = (ref["is_vals"][i] - vec["is_vals_all"][i]).abs().max().item()
        if di != 0 or dv > 1e-4:
            is_ok = False
            print(f"  FAIL IS row {i}: dids={di} dvals={dv:.2e}")
    print(f"  {'OK ' if is_ok else 'FAIL'} IS top-k (triggered rows)")
    allok &= is_ok
    print("\n==>", "ALL EQUIVALENT" if allok else "MISMATCH — do not ship")


if __name__ == "__main__":
    main()
