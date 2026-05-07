# Mechanistic Interpretability: Reading List

A curated reading list for understanding the mechanistic interpretability literature, particularly relevant to positional bifurcation, representation divergence, and the PIVOT co-design hypothesis.

## Priority Reading Order

Start here if new to the field: **1 → 2 → 4 → 6 → 7**

---

## Foundations

### 1. A Mathematical Framework for Transformer Circuits
**Elhage et al., Anthropic, 2021**
- Introduces the "circuits" lens: residual stream, attention heads, MLP layers as composable primitives
- Defines key concepts: superposition, virtual attention heads, QK/OV decomposition
- **Why read**: The vocabulary for all subsequent work. PIVOT's attention-hook analysis maps directly onto this framework.

### 2. In-context Learning and Induction Heads
**Olsson et al., Anthropic, 2022**
- Identifies induction heads as the mechanistic substrate of in-context learning
- Shows phase transitions during training corresponding to capability jumps
- **Why read**: Phase transitions ↔ positional bifurcation. The idea that a specific circuit activates at a specific position mirrors our fork-position hypothesis.

### 3. Toy Models of Superposition
**Elhage et al., Anthropic, 2022**
- Explains how models pack more features than dimensions into the residual stream via superposition
- Shows geometry of feature directions in representation space
- **Why read**: Background for understanding why `v[t] = mean(h_t|correct) - mean(h_t|wrong)` has a meaningful direction in high-dimensional space.

---

## Representation-Level Methods

### 4. Representation Engineering: A Top-Down Approach to AI Transparency
**Zou et al., 2023** — [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- Extracts "representation reading vectors" from contrastive pairs (e.g., honest vs. dishonest)
- Shows that linear directions in hidden states correspond to high-level concepts
- **Why read**: PIVOT-v2's fork score `v[t] = mean(h_t|correct) - mean(h_t|wrong)` is exactly a representation engineering direction, computed per position. This paper validates the approach theoretically and empirically.

### 5. Steering Llama 2 via Contrastive Activation Addition
**Panickssery et al., 2024** — [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
- Adds contrastive direction vectors to residual stream to steer behavior at inference
- Extends Representation Engineering to generation-time control
- **Why read**: The Langevin perturbation in PIVOT-v2 is a stochastic version of activation addition — nudging the model at fork positions. This paper provides the deterministic analogue.

---

## Circuit-Level Analysis

### 6. Interpretability in the Wild: A Circuit for Indirect Object Identification
**Wang et al., 2022** — [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Full circuit dissection for a specific task (IOI) in GPT-2
- Identifies name-mover heads, duplicate token heads, inhibition heads
- **Why read**: Shows the methodology for tracing which attention heads contribute to positional decisions. Directly applicable to identifying which heads drive bifurcation.

### 7. Locating and Editing Factual Associations in GPT (ROME)
**Meng et al., 2022** — [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
- Uses causal interventions (activation patching) to localize factual knowledge to specific MLP layers and token positions
- Introduces rank-one model editing
- **Why read**: Causal mediation analysis is the right tool for confirming which positions are causally responsible for correct vs. wrong rollout divergence. Directly relevant to validating the bifurcation hypothesis.

---

## RL / Training Dynamics

### 8. Progress Measures for Grokking via Mechanistic Interpretability
**Nanda et al., 2023** — [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Tracks circuit formation during training, connects grokking (generalization phase transition) to circuit emergence
- **Why read**: RL training on math problems may exhibit similar phase transitions. The v6 AIME plateau-then-kickin pattern could be a grokking-like transition at fork positions.

---

## Scaling / Feature Dictionaries

### 9. Towards Monosemanticity + Scaling Monosemanticity
**Anthropic, 2023–2024**
- Uses sparse autoencoders (SAEs) to decompose superposed MLP activations into monosemantic features
- Towards: [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features)
- Scaling: [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity)
- **Why read**: If fork positions correspond to specific feature directions, SAEs could identify which features activate differentially at those positions — a path to mechanistic validation of PIVOT.

---

## Connections to PIVOT

| Paper | Connects to PIVOT via |
|---|---|
| Circuits framework (1) | Hook architecture, residual stream analysis |
| Induction heads (2) | Phase transitions at specific positions |
| Superposition (3) | Why `v[t]` is meaningful in high-d |
| Representation Engineering (4) | **Direct basis for PIVOT-v2 Phase 1** |
| Activation Addition (5) | Deterministic analogue of Langevin nudge |
| IOI circuit (6) | Attention-head attribution at fork positions |
| ROME (7) | Causal validation of bifurcation positions |
| Grokking (8) | RL training dynamics, plateau-then-kickin |
| Monosemanticity (9) | Feature-level explanation of fork directions |

---

## Suggested Experiments Enabled by This Literature

1. **Activation patching (ROME methodology)**: patch `h_t` from correct → wrong rollout at candidate fork position `t*`. If patching reverses the outcome, `t*` is causally responsible.
2. **Representation reading vectors**: plot `||v[t]||` across positions for math problems grouped by difficulty. Do hard problems have earlier, sharper forks?
3. **Attention head attribution**: at fork positions, which heads change most between correct/wrong rollouts? Are they consistent across prompts?
4. **SAE feature activation at fork positions**: which features in a trained SAE activate differentially at `t*`? Do they correspond to "planning" or "uncertainty" features?
