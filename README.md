<div align="center">

# DRIFT — Directed Rollout with Inferred Feedback at Triggers

**Learned Langevin exploration on the logit manifold for reasoning RL**

</div>

This repository implements **DRIFT**, an early-stage exploration-driven optimization
module for GRPO-style RL on LLM reasoning. DRIFT injects a learned Langevin
update at high-entropy *branch points* during rollout to break the *dead zone*
— the regime where every rollout in a GRPO group receives the same reward and
the policy gradient identically vanishes — without paying the entropy-collapse
or ceiling-degradation cost of a uniform entropy bonus.

The method paper is [docs/drift_method.md](docs/drift_method.md). The
implementation is built on top of [verl](https://github.com/volcengine/verl),
the Volcano Engine RL training library.

---

## What DRIFT does

GRPO sets the policy gradient using within-group reward variance across `n`
rollouts. On hard reasoning tasks, a base model's all-incorrect rate is so
close to 1 that almost every group of `n=8` rollouts is uniformly wrong —
the within-group standard deviation collapses to zero, the advantage is
identically zero, and no parameter movement occurs. On Qwen3-4B-Base trained
on competition mathematics, vanilla GRPO produces no measurable validation
signal for roughly the first 100 training steps.

DRIFT attacks this regime with three coupled components:

- **Branch-point selection.** A token position is a branch point only if its
  next-token entropy exceeds an adaptive (top-15%) quantile threshold *and*
  it lies past a positional guard `t_min`. The guard prevents Langevin
  perturbation from being exploited by a short-answer shortcut at early
  positions.

- **Learned Langevin rollout.** At each branch point, we apply
  `l̃_t = l_t + η·ε_t + σ·ζ_t` where `ε_t` mixes a learned drift `G_t` with
  isotropic exploration on the unit sphere, and `ζ_t` is Gaussian diffusion.
  The update is confined to the top-K logit coordinates so the perturbation
  redistributes mass among the model's existing high-probability candidates
  rather than activating long-tail noise. The diffusion term is what
  spreads `n` rollouts of a GRPO group into distinct continuations,
  restoring the within-group reward variance GRPO needs.

- **Entropy-maintenance feedback.** The drift `G_t` is updated with a
  Polyak-filtered SPSA estimate. Crucially, the feedback signal is
  `s_t = H_t − α_target · H_first` (entropy *maintenance*), not
  `s_t = H_prev − H_t` (entropy *reduction*). The dead zone is a regime of
  confidently-wrong rollouts; rewarding entropy reduction reinforces that
  failure mode. Maintenance instead reinforces perturbations that keep
  later branch points perturbable, so subsequent Langevin updates retain
  their ability to redirect the trajectory.

A length-conditioned score adjustment (correctness-conditional, applied
pre-normalization) suppresses a short-correct-answer shortcut that the
Langevin perturbation can otherwise exploit. Off-policy trajectories are
incorporated into GRPO with a bounded mixture-denominator importance
correction (Kakade & Langford, 2002).

The full method, theoretical analysis (branch-point discovery
amplification), and implementation details are in
[docs/drift_method.md](docs/drift_method.md).

---

## Repository layout

DRIFT-specific code (everything else is upstream verl):

| Path | What it is |
|---|---|
| [verl/utils/vllm/pivot_patch.py](verl/utils/vllm/pivot_patch.py) | The Langevin rollout patch for vLLM. Implements branch-point selection, top-K perturbation, SPSA drift estimation, and the entropy-maintenance feedback. |
| [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) | GRPO advantage with the length-conditioned score adjustment (correctness-conditional, pre-normalization). |
| [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) | Trainer integration; off-policy IS correction with mixture denominator at trigger positions. |
| [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py) | Per-token clipping and asymmetric ε_low / ε_high for the corrected ratio. |
| [examples/grpo_trainer/run_qwen3_*_pivot_v18*.sh](examples/grpo_trainer/) | Training scripts for the headline DRIFT runs (1.7B and 4B). |
| [examples/grpo_trainer/run_qwen3_*_grpo*.sh](examples/grpo_trainer/) | GRPO and HEG (high-entropy GRPO) baselines, same data and hyperparameters except for the exploration mechanism. |
| [examples/grpo_trainer/guru_rl_reward.py](examples/grpo_trainer/guru_rl_reward.py) | Custom reward routing — `math__*` data sources go to `math_dapo` answer-equivalence, `mcq__*` go to MCQ regex. |
| [examples/data_preprocess/prepare_eval_*.py](examples/data_preprocess/) | Eval-set prep for AIME24/AIME25/OlympiadBench/GPQA/Minerva. |
| [examples/eval/](examples/eval/) | End-to-end eval pipeline: per-method scripts (`eval_v18b_1p7b.sh`, etc.), aggregator producing markdown tables for the paper. |
| [examples/empirical/](examples/empirical/) | Three empirical mechanism studies: rollout diversity, reward variance over training, branch-point entropy time-series. |
| [docs/drift_method.md](docs/drift_method.md) | Method paper (algorithm, theorems, hyperparameter table). |
| [docs/experiment_todo.md](docs/experiment_todo.md) | Tracking checklist for the experiment section. |

---

## Quick start

### Install

Follow the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html).
DRIFT requires no additional dependencies beyond verl + vLLM + FSDP.

### Reproduce the headline result

```bash
# 1. Prepare eval datasets (AIME25, OlympiadBench, GPQA, Minerva)
cd examples/data_preprocess
bash prepare_all_eval.sh

# 2. Train DRIFT on Qwen3-4B-Base
cd ../grpo_trainer
bash run_qwen3_4b_base_pivot_v18c.sh

# 3. Train baselines for comparison
bash run_qwen3_4b_base_grpo.sh           # GRPO
bash run_qwen3_4b_base_high_ent_grpo.sh  # HEG (uniform entropy bonus)

# 4. Evaluate saved checkpoints across the benchmark suite
cd ../eval
bash eval_v18c_4b.sh
bash eval_grpo_4b.sh
bash eval_heg_4b.sh

# 5. Aggregate into doc-ready markdown tables
python3 aggregate_eval.py \
    --root /path/to/eval_out \
    --format md > tables.md
```

### Mechanism studies

```bash
cd examples/empirical

# Reward variance over training (no compute, parses training JSONLs):
python3 reward_variance.py \
    --jsonls GRPO=/path/to/grpo.jsonl HEG=/path/to/heg.jsonl DRIFT=/path/to/drift.jsonl \
    --out reward_variance.json

# Rollout diversity within a GRPO group (one inference pass per checkpoint):
bash run_diversity_sweep.sh

# Branch-point entropy time-series (single-GPU, ~5 min per prompt):
python3 entropy_time_series.py --ckpt /path/to/drift_ckpt --tag drift_v18b_step120
```

See [examples/empirical/README.md](examples/empirical/README.md) for what each
study defends and how to plot the outputs.

---

## Hyperparameters (defaults, see §5 of the paper)

| Component | Symbol | Value |
|---|---|---|
| Branch-point quantile | p | 0.85 |
| Positional guard | t_min | 800 |
| Top-K subspace | K | 20 |
| Drift step size | η | 0.1 |
| Diffusion magnitude | σ | 0.01 |
| Polyak momentum | γ | 0.7 |
| Drift exploit ratio | α | 0.6 |
| Entropy-maintenance target | α_target | 0.7 |
| Length penalty coefficient | λ | 2.0 |
| Target reasoning length | L_target | 1000 |
| IS mixture coefficient | β | 0.5 |
| Entropy cap | H_cap | 0.8 |
| Asymmetric clip range | (ε_low, ε_high) | (0.2, 0.28) |
| KL coefficient | λ_KL | 0.001 |

---

## Citation

(TODO: add bibtex once the paper is on arXiv.)

This work builds on verl. If you use this codebase, please also cite verl:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv:2409.19256}
}
```

---

## Acknowledgements

DRIFT is implemented on top of [verl](https://github.com/volcengine/verl)
(Volcano Engine RL Training Library), which provides the Ray + FSDP + vLLM
hybrid-controller infrastructure. The dead-zone framing and Langevin formulation
are independent contributions; the reward-routing, agent-loop, and
reward-manager hooks are minimal extensions of verl's existing pipeline. We
thank the verl maintainers and the open-source community for the underlying
training framework.
