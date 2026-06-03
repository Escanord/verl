# Compute Scaling: 4B → 30B for DRIFT Training

**Goal.** Estimate the cluster size required to train a dense 30B model with DRIFT (full-parameter RL, FSDP actor + colocated vLLM rollout) under the constraint that all training hyperparameters from the current 4B run are preserved.

**Source of truth.** The current 4B run is `examples/grpo_trainer/run_qwen3_4b_base_pivot_v18c.sh` on Qwen3-4B-Base, 8×H200 (143 GB/GPU), with `ppo_micro_batch_size_per_gpu=16`, `ppo_mini_batch_size=256`, `train_batch_size=1024`, `rollout.n=8`, `max_response_length=4096`, `rollout.max_num_seqs=512`, `gpu_memory_utilization=0.4`. No LoRA, no activation offload, no Megatron — pure FSDP.

**Observed peak.** ~120 GB/GPU on the 4B run.

---

## 1. Memory Decomposition of the 4B Run

The 120 GB/GPU peak under the current config breaks down approximately as follows:

| Component | 4B estimate | Scales with |
|---|---:|---|
| FSDP actor state: params (bf16) + grads (bf16) + master (fp32) + AdamW m,v (fp32) = 16·P/N | ~8 GB | 1/N |
| vLLM allocation (`gpu_memory_utilization=0.4`): bf16 weights + KV cache | ~57 GB | 1/TP for weights, 1/TP for sharded KV |
| PPO backward activations (saved per-layer residuals under grad checkpointing) | ~14 GB | model arch × micro-batch — **fixed in N** |
| Logits buffer at log-prob and update (micro_batch × seq × vocab × bf16) | ~24 GB | micro-batch — **fixed in N** |
| FSDP all-gather transient + Adam scratch + reference model wake | ~15 GB | partly 1/N |
| **Total** | **~118 GB** | |

The two italicised rows — activations and the logits buffer — are the load-bearing constraint for the 30B scaling question. Under pure FSDP, neither shrinks when GPUs are added; only sequence parallelism or pipeline parallelism would shard them.

---

## 2. Scaling Multipliers to a Dense 30B

Reference architecture: dense 30B-class (Qwen2.5-32B / QwQ-32B class) — 64 layers, hidden 5120, GQA with 8 KV heads, vocab ~152K.

| Component | Multiplier 4B → 30B | Per-GPU 30B estimate |
|---|---:|---:|
| FSDP state | × P = 7.5 | **480 / N GB** |
| Activations at preserved micro-batch | × hidden² × layers ≈ 4× | **~55–60 GB** |
| Logits buffer | × 1 (vocab nearly identical) | **~24 GB** |
| vLLM weights | × P / TP | **60 / TP GB** |
| vLLM KV cache | × (layers × d_kv) ≈ 3× | **~150 / TP GB** |
| Misc (all-gather, Adam scratch, ref wake) | partly × P / N | **~10–15 GB** |

**Critical observation.** With micro-batch and batch size preserved, activations + logits + misc set a **fixed floor of ~90 GB/GPU** that is independent of cluster size under pure FSDP. The H200's 143 GB cap leaves ~53 GB per GPU for FSDP state and vLLM combined, which dictates the minimum cluster size.

---

## 3. Per-GPU Peak by Cluster Size

vLLM TP fixed at 8 in every row below (necessary so weights fit at any scale; ~26 GB/GPU for weights + sharded KV).

| Cluster size | N (GPUs) | FSDP state | vLLM (TP=8) | Activations + logits + misc | **Total peak / GPU** | Fits 143 GB? |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 cluster | 8 | 60 GB | 26 GB | 90 GB | **176 GB** | No |
| 2 clusters | 16 | 30 GB | 26 GB | 90 GB | **146 GB** | No (3 GB over) |
| 3 clusters | 24 | 20 GB | 26 GB | 90 GB | **136 GB** | Tight (7 GB margin) |
| **4 clusters** | **32** | **15 GB** | **26 GB** | **90 GB** | **131 GB** | **Yes (12 GB margin)** |
| 6 clusters | 48 | 10 GB | 26 GB | 90 GB | **126 GB** | Yes (17 GB margin) |
| 8 clusters | 64 | 7.5 GB | 26 GB | 90 GB | **123 GB** | Yes (20 GB margin) |

---

## 4. Recommendation

| Option | Clusters (8×H200) | Total GPUs | Per-GPU peak | Margin | Verdict |
|---|---:|---:|---:|---:|---|
| Minimum | 4 | 32 | ~131 GB | 12 GB | Feasible, vulnerable to OOM during length spikes. |
| **Recommended** | **6** | **48** | **~126 GB** | **17 GB** | Survives mid-training response-length growth (~50% drift observed empirically in PIVOT-v2, v5a). |
| Comfortable | 8 | 64 | ~123 GB | 20 GB | Standard published-recipe territory for 30B full-FT RL. |

The 12–20 GB margin is not slack — response length on math RL runs has been observed to drift up by ~50% during training (PIVOT-v2 N=20→145: 677 → 1215 tokens; PIVOT-v5a: 994 → 1215 tokens), which pushes both activation memory and KV-cache demand up proportionally. The 4-cluster minimum is the right number on a model-arch spreadsheet; the 6-cluster recommendation is the right number for a run that does not require step-by-step memory babysitting.

---

## 5. Summary for Team

| Question | Answer |
|---|---|
| Can we train DRIFT on a dense 30B with one 8×H200 cluster? | No. Peak per-GPU memory exceeds H200's 143 GB by ~33 GB under preserved config. |
| What is the minimum cluster count? | **4 clusters (32 H200s)** with pure FSDP and preserved config. |
| What is the recommended cluster count? | **6 clusters (48 H200s)** to survive response-length drift without hand-tuning. |
