---
title: "7.4 1F1B and DualPipe"
chapter_title: "Computational Performance and AI Infra"
section_id: "07-04"
language: en
source_language: zh
source_docx: "第1部分 深度学习/7.计算性能与AI Infra/7.4 1F1B与DualPipe.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 7.4 1F1B and DualPipe

## I. Basic Training Steps in Distributed Training Frameworks

1. F (Forward): forward computation

Action: input data $x$ pass through the current layer's operator (such as matrix multiplication $Wx$) to produce activations $y$.

Formula:

$$
y = f(x, \theta)
$$

Key task: compute predictions and **cache intermediate-layer activations through checkpointing** for subsequent backpropagation.

In pipeline parallelism: F's output must be sent to the next, downstream GPU node.

2. B (Backward for Input / Activation Gradient): activation-gradient computation

Action: use the gradient $\frac{\partial L}{\partial y}$ returned from the preceding layer to compute the gradient with respect to the **current layer's input** $x$.

Formula:

$$
\nabla x = \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

Key task: this is the “baton” of backpropagation. The computed $\nabla x$ must immediately be sent back to the previous, upstream GPU node so that its layer can continue computing.

Role: in pipeline parallelism, B lies on the **critical path**. If B computes slowly or its transfer latency is high, the entire pipeline stalls (creating a bubble).

3. W (Backward for Weights / Parameter Gradient): weight-gradient computation

Action: use the gradient $\frac{\partial L}{\partial y}$ to compute the gradient with respect to the **current layer's parameters** $\theta$.

Formula:

$$
\nabla \theta = \frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial \theta}
$$

Key task: the computed $\nabla \theta$ is stored in a gradient buffer and eventually used by the optimizer to update model weights.

Role: $W$ **is not on the critical path**, because $\nabla \theta$ is needed only for a local update at this layer and does not have to be passed to other ranks to resolve computational dependencies.

## II. 1F1B

Suppose we have four GPUs (ranks 0–3) and pipeline depth $PP = 4$.

- **For rank 0**: forward computation ($F_1$) for microbatch $m_1$ is very fast.
- **However**: after $F_1$ is computed, it must pass to rank 1 $\to$ rank 2 $\to$ rank 3. Once rank 3 computes the loss, gradients travel from rank 3 $\to$ rank 2 $\to$ rank 1 before finally returning to rank 0.
- **Conclusion**: a long time has elapsed by the time rank 0 receives the gradient for $m_1$ and is ready to compute its backward pass ($B_1$).

To keep the GPU busy while waiting for the gradient for $B_1$, rank 0 continues computing $F_2, F_3, F_4, \dots$.

- $F_i$: the forward pass of the current, $i$-th microbatch.
- $B_{i-k}$: the backward pass of an earlier microbatch (the $i-k$-th) that **has already completed a full round trip**.

Here, $k$ actually depends on the GPU's rank position.

- At the end of the pipeline (such as rank 3), $F$ and $B$ are close together, so $k$ is small.
- At the beginning of the pipeline (such as rank 0), $F$ and $B$ are far apart, so $k$ is large.

“Bubble” size:

Suppose each stage ($F$ or $B$) takes one unit of time:

1. $T = 0$: rank 0 executes $F_1$.
2. $T = PP - 1$: after $PP - 1$ hops, $F_1$ reaches the last rank and completes computation; the loss is computed immediately.
3. $T = PP$: the last rank begins backward computation $B_1$.
4. $T = PP + (PP - 1)$: after $PP - 1$ hops, the gradient returns to rank 0.
5. $T = 2PP - 1$: rank 0 can finally begin $B_1$.

Conclusion: from rank 0's perspective, $2PP - 2$ time slots separate the end of $F_1$ and the start of $B_1$.

A conceptual distinction is needed here: **single-task latency** versus **total pipeline bubbles**.

- **Total bubbles**: the total time that all GPUs spend idle during the entire training task.
- In 1F1B, rank 0 does not remain idle while waiting for $B_1$ to return; it rapidly executes $F_2, F_3, \dots, F_{PP}$.
- The only idle time that truly cannot be hidden is the initial “warm-up” and final “cool-down.” Summing all GPU idle time and averaging it gives an average bubble size per GPU of approximately $(PP - 1) \times (F + B)$.

## III. DualPipe

We use **four GPUs (ranks 0–3)** and pipeline parallel depth $PP = 4$ as an example.

- **Forward-direction flow (D1)**: data travel $R0 \to R1 \to R2 \to R3$ (logical layers L1 $\to$ L8).
- **Reverse-direction flow (D2)**: data travel $R3 \to R2 \to R1 \to R0$ (logical layers L1 $\to$ L8; note the mirrored arrangement).
- **Computation splitting**: each stage is divided into $F$ (forward), $B$ (input gradients, sent to a neighbor), and $W$ (weight gradients, used locally).

### (1) Bidirectional Warm-up

**$T1$: both directions start simultaneously**

- Rank 0: $F_{D1}(m_1)$
- Rank 1: -
- Rank 2: -
- Rank 3: $F_{D2}(n_1)$

**$T2$: data converge toward the middle**

- Rank 0: $F_{D1}(m_2)$
- Rank 1: $F_{D1}(m_1)$
- Rank 2: $F_{D2}(n_1)$
- Rank 3: $F_{D2}(n_2)$

**$T3$: middle nodes begin handling both flows**

- Rank 0: $F_{D1}(m_3)$
- Rank 1: $F_{D1}(m_2)+F_{D2}(n_1)$
- Rank 2: $F_{D2}(n_2)+F_{D1}(m_1)$
- Rank 3: $F_{D2}(n_3)$

**$T4$: the first batches of both flows reach the ends**

- Rank 0: $F_{D1}(m_4)+F_{D2}(n_1)$
- Rank 1: $F_{D1}(m_3)+F_{D2}(n_2)$
- Rank 2: $F_{D2}(n_3)+F_{D1}(m_2)$
- Rank 3: $F_{D2}(n_4)+F_{D1}(m_1)$
- Note: $m_1$ has now reached $R3$, and $n_1$ has reached $R0$.

Analysis: at the end of $T4$, $m_1$ has completed its entire forward pass (logical L1–L8) at $R3$, and $n_1$ has completed its entire forward pass at $R0$. Starting at $T5$, the system produces its first gradients and enters the complex overlapping phase.

Note: m1 and n1 are completely different data batches.

### (2) Steady State

In steady state, a GPU does not execute sequentially; instead, it works through two main **parallel candidate blocks**. A complete steady-state step contains the following two central overlaps:

**Overlap A: complete overlap of $F_{D1}$ and $B_{D2}$**

- **Computation**: the GPU launches two operators simultaneously:
  1. Forward computation $F_{D1}(m_i)$ for a new batch in the forward-direction flow.
  2. Activation-gradient computation $B_{D2}(n_j)$ for an older batch in the reverse-direction flow.
- **Communication hiding**:
  - While computing $F_{D1}$, receive downstream gradients needed by $B_{D2}$ in the background.
  - While computing $B_{D2}$, send activations produced by $F_{D1}$ in the background.
- **Conclusion**: as long as computation times satisfy $T(F) \approx T(B)$, the two operators can completely hide each other's communication time.

**Overlap B: complete overlap of $F_{D2}$ and $B_{D1}$**

- Mirroring the logic above, this handles forward computation of the reverse-direction flow and activation gradients of the forward-direction flow.

**Step C: filling gaps with $W$ (weight gradients)**

- **Computation**: immediately after the above $B$ computation finishes, compute $W$ (weight gradients) asynchronously in the background.
- **Principle**: $W$ affects no one else. It is “leftover computation” that can fill any available gap.

An overlapping cycle (such as the overlap of $F_{D1}$ and $B_{D2}$) contains four communication steps, all hidden within computation time:

While computing $F_{D1}$:

- `Send-Backward-D2`: send the previously computed gradient for $n_{j-1}$ to the right.

- `Recv-Backward-D2`: receive the current batch's gradient for $n_j$ from the left.

While computing $B_{D2}$:

- `Send-Forward-D1`: send the computed activations for $m_i$ to the right.

- `Recv-Forward-D1`: receive the next batch's activations for $m_{i+1}$ from the left.

### (3) Cool-down

Once all microbatches have completed their forward passes, GPUs enter backward-only mode.

1. **Process the remaining $B$ operations**: finish computing and sending input gradients for the last few microbatches. No $F$ operations remain to overlap, but because $PP$ is divided into two directions, the absolute bubble duration is halved.
2. **Compute $W$ at full capacity**: this is the final heavy lifting; all GPUs compute weight updates for their assigned layers in parallel.

### (4) Core Reasons for Bubble Reduction

In 1F1B, after rank 0 computes $F_1$, it waits $2(PP - 1)$ time slots before receiving the gradient needed to compute $B_1$.

$$
T_{\mathrm{wait}} = (PP - 1) \times (F + B)
$$

DualPipe uses the following approaches:

1. **Bidirectional overlap**:
   - While 1F1B waits for gradients, DualPipe has the GPU run forward and backward computations in the “other direction.”
   - **Mathematical logic**: divide one PP chain into two half-depth PP chains that overlap. The bubble changes from $(PP - 1)$ to $\left(\frac{PP}{2} - 1\right)$.
2. **Splitting $B$ and $W$ (the Zero Bubble idea)**:
   - Traditionally, backpropagation consists of $B + W$. Both must finish before gradients can be sent to the previous GPU.
   - DualPipe **sends gradients immediately** after computing $B$ (requiring only half the matrix-multiplication workload), allowing the neighboring GPU to start working earlier.
   - **Result**: waiting time on the critical path is further reduced.
3. **Full overlap**:
   - Exploit modern GPUs' duplex bandwidth (NVLink can send and receive simultaneously at full speed).
   - DualPipe's scheduling algorithm guarantees that **bandwidth is always fully utilized while the GPU computes, and SM cores are always computing while data are being transferred.**

### (5) Folded Pipeline Technique

The “cost” of DualPipe is that each GPU must store twice the weights. The reason is:

> Forward-direction flow, Direction 1: from $Rank_0 \rightarrow Rank_3$. Rank 0 must have layer 1.
>
> Reverse-direction flow, Direction 2: from $Rank_3 \rightarrow Rank_0$. Note that the reverse-direction flow is also a complete forward-backward process. If it starts at rank 3, rank 3 must have layer 1.
>
> Thus layer 1 exists on both rank 0 and rank 3, naturally doubling the parameter count.
To address this, a “folded” approach can be used:

In the folded architecture, rather than storing layer 1 on both rank 0 and rank 3, stages are assigned as follows:

| Physical rank | Assigned logical stages (1x parameters, no redundancy) | Physical location |
| --- | --- | --- |
| Rank 0 | Stage 0 (entry) and stage 7 (exit) | Both ends of the pipeline |
| Rank 1 | Stage 1 and stage 6 |  |
| Rank 2 | Stage 2 and stage 5 |  |
| Rank 3 | Stage 3 and stage 4 | Pipeline turning point |

Although each stage is stored only once (1x parameters), every GPU now holds both an early stage (such as S0) and a late stage (such as S7), so overlap remains possible:

1. Task combination: while rank 0 processes stage 0 (forward computation) for microbatch $m_{10}$, it may also hold stage 7 (forward or backward computation) for microbatch $m_1$, which has completed a round trip.
2. Operator overlap:
   - Computation A: $F_{S_0}$ (forward pass of a new batch).
   - Computation B: $B_{S_7}$ (backward pass of an older batch).
3. Communication hiding:
   - While rank 0 sends $F_{S_0}$ to rank 1, it simultaneously receives the $B_{S_7}$ gradient returned by rank 1.

This is the essence of “folding”: pipeline depth (a microbatch takes a long time to travel from S0 to S7) creates a time offset, allowing the same GPU to handle tasks at different lifecycle stages simultaneously.
In practice, however, many large models do not necessarily use folding, for two reasons.

1. Characteristics of MoE models

In a very large MoE model, parameters are divided into two parts:

- **Shared parameters**: attention, LayerNorm, and some shared dense layers.
- **Expert parameters**: hundreds or thousands of independent experts (MLP blocks) in MoE layers.

The central implementation detail is that **expert parameters do not participate in pipeline-mirroring redundancy.**

Distributed training usually combines PP (pipeline parallelism) and EP (expert parallelism):

1. **PP determines the route**: it determines which GPUs a microbatch travels between.
2. **EP determines storage**: it determines which physical device memory holds a specific expert $E_{100}$.

In DualPipe's bidirectional flows:

- When a microbatch reaches a layer that needs an expert, it initiates All-to-All communication to locate the physical GPU hosting that expert.
- Whether the microbatch belongs to the “forward-direction flow” or “reverse-direction flow,” it seeks the expert at the same physical address.
- **Conclusion:** expert parameters are still stored only once across the entire cluster. DualPipe merely needs an extra copy of lightweight shared-layer parameters and router logic on every rank.

2. Activations occupy more device memory

For extremely long-context training, the device-memory bottleneck is often activations rather than weights.

- During training, every layer's forward results must be stored for backpropagation.
- DualPipe's memory pressure: what truly strains memory is running flows in both directions simultaneously, requiring activations for both $m$ and $n$ batches to be stored at once.

DeepSeek cleverly mitigates this:

- Multi-head Latent Attention (MLA) substantially compresses the KV cache and activations.
- Activation recomputation can recompute some activations when needed, trading compute time for device-memory capacity.
- By comparison, the few GB occupied by the extra shared-parameter copy are “a drop in the bucket.”

3. Training scheduling

Extracting maximum bandwidth:

A folded pipeline with 1x parameters is highly rigid in scheduling (it must follow a V-shaped route). DualPipe with 2x parameters has two completely independent, symmetric paths, allowing it to fully saturate bidirectional NVLink bandwidth with extremely flexible scheduling.

Pursuing the minimum bubble rate:

In the folded model, more complex dependencies between $B$ and $W$ make it difficult to achieve nearly “zero bubbles” as DualPipe does. DeepSeek aims to push model FLOPs utilization (MFU) above 70%, making 2x redundancy a worthwhile trade-off.

## References

- Narayanan, D., Harlap, A., Phanishayee, A., et al. (2019). [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://dl.acm.org/doi/10.1145/3341301.3359646). SOSP 2019.
- DeepSeek-AI. (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). arXiv:2412.19437.
