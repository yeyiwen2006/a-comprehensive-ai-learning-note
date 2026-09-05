---
title: "19.2 Flash Attention"
chapter_title: "Engineering Optimizations for Attention"
section_id: "19-02"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.2 Flash Attention.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.2 Flash Attention

## I. The Core Problem Addressed by Flash Attention

If KV cache addresses the time-complexity problem during inference, the core purpose of Flash Attention is to address the space-complexity problem during training, that is, the memory-bound I/O bottleneck.

In a GPU, the computational unit (SRAM) is extremely fast but has a small capacity, whereas GPU memory (HBM) has a large capacity but slow reads and writes. During standard attention computation, the intermediate $N \times N$ matrices are too large to fit in SRAM, requiring frequent reads from and writes to HBM.

Flash Attention uses two core mathematical techniques to achieve acceleration and GPU-memory savings: tiling and recomputation.

The input matrices are $Q,K,V \in \mathbb{R}^{N \times d}$. Standard attention calculates the output $O$ as:

$$
S = QK^T,\quad S \in \mathbb{R}^{N \times N}
$$

$$
P = \mathrm{softmax}(S),\quad P \in \mathbb{R}^{N \times N}
$$

$$
O = PV,\quad O \in \mathbb{R}^{N \times d}
$$

The key problem is that both $S$ and $P$ are $N \times N$ matrices. When the sequence length $N$ is large, their GPU-memory consumption grows with $N^2$, becoming the main bottleneck in training long-sequence models.

## II. Tiling and Online Softmax

Given sequence length $N$, attention-head dimension $d$, and GPU on-chip SRAM capacity $M$ (measured in numbers of floating-point elements), the paper calculates the tile sizes as follows to ensure that subblocks of $Q$, $K$, and $V$ fit in SRAM during computation:

- Column tile size (corresponding to the sequence dimension of $K,V$):

$$
B_c = \left\lceil \frac{M}{4d} \right\rceil
$$

- Row tile size (corresponding to the sequence dimension of $Q$):

$$
B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)
$$

Based on these two tile sizes, the input matrices are partitioned as follows:

- $Q \in \mathbb{R}^{N \times d}$ is divided into $T_r = \left\lceil \frac{N}{B_r} \right\rceil$ blocks, $Q_1,\ldots,Q_{T_r}$, each of size $B_r \times d$.
- $K,V \in \mathbb{R}^{N \times d}$ are each divided into $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ blocks, $K_1,\ldots,K_{T_c}$ and $V_1,\ldots,V_{T_c}$, each of size $B_c \times d$.
- The output matrix $O \in \mathbb{R}^{N \times d}$ and statistic vectors $l \in \mathbb{R}^N$, $m \in \mathbb{R}^N$ are also partitioned by rows into $T_r$ blocks.

Note: a token's $q$ vector is not split across different blocks; the same applies to $k$ and $v$. In other words, tiling occurs along the sequence dimension, not the attention-head dimension. Every pair $Q_i$ and $K_j$ is multiplied to obtain a local attention score block:

$$
S_{ij} = Q_iK_j^T
$$

All $S_{ij}$ together cover the complete $S = QK^T$. The output is not obtained by simply concatenating block results. Instead, with corrections using online Softmax statistics, the $PV$ contribution of each local block is accumulated into the current output block.

To avoid generating a huge $N \times N$ matrix all at once, Flash Attention partitions the inputs $Q,K,V$ into small blocks and computes them blockwise in SRAM.

The difficulty is that Softmax is a global operation, depending on the maximum and denominator sum of an entire row:

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\quad m = \max(x)
$$

Looking at only part of the data cannot determine the global $m$ and denominator. Flash Attention therefore uses the online Softmax algorithm to update local statistics dynamically as new blocks are processed.

Suppose we process two blocks. The first block has local maximum $m_1$, local exponential sum $l_1$, and unnormalized output $O_1$; the second has $m_2,l_2,O_2$. The merged global maximum $m_{new}$ and global exponential sum $l_{new}$ are updated as follows:

$$
m_{new} = \max(m_1,m_2)
$$

$$
l_{new} = e^{m_1-m_{new}}l_1 + e^{m_2-m_{new}}l_2
$$

The final output $O_{new}$ is updated as:

$$
O_{new} = \mathrm{diag}(l_{new})^{-1}\left(\mathrm{diag}\left(l_1 \odot e^{m_1-m_{new}}\right)O_1 + \mathrm{diag}\left(l_2 \odot e^{m_2-m_{new}}\right)O_2\right)
$$

In this way, Flash Attention only needs to traverse $K,V$ once, continuously updating $O$ in SRAM before finally writing it back to HBM. The intermediate $N \times N$ attention matrices $S$ and $P$ never exist in full in HBM.

### Algorithm Workflow

The outer loop traverses query blocks. For $i = 1$ to $T_r$:

1. Load $Q_i$ from HBM into SRAM.
2. Initialize local accumulators for the current block in SRAM:

$$
O_i = 0,\quad l_i = 0,\quad m_i = -\infty
$$

3. The inner loop traverses key/value blocks. For $j = 1$ to $T_c$:

   1. Load $K_j$ and $V_j$ from HBM into SRAM.
   2. Calculate the local attention scores by matrix multiplication in SRAM:

$$
S_{ij} = Q_iK_j^T \in \mathbb{R}^{B_r \times B_c}
$$

   3. Calculate the local maxima:

$$
m_{ij} = \mathrm{rowmax}(S_{ij})
$$

   4. Update the global maxima by comparing the old row maxima with the newly calculated local maxima:

$$
m_i^{new} = \max(m_i,m_{ij})
$$

   5. Calculate the local unnormalized exponential weights:

$$
P_{ij} = e^{S_{ij}-m_i^{new}}
$$

   6. Update the global exponential sums with scaling compensation:

$$
l_i^{new} = e^{m_i-m_i^{new}}l_i + \sum_{\text{row}} P_{ij}
$$

   7. Update the output matrix block with scaling compensation and accumulation of new values:

$$
O_i^{new} = e^{m_i-m_i^{new}}O_i + P_{ij}V_j
$$

   8. Update the state, replacing the current state with the new global state in preparation for the next block $j$:

$$
m_i \leftarrow m_i^{new},\quad l_i \leftarrow l_i^{new},\quad O_i \leftarrow O_i^{new}
$$

4. When the inner loop ends, meaning that the current $Q_i$ has been processed with all $K,V$ blocks, apply the actual Softmax normalization to the accumulated $O_i$:

$$
O_i = \mathrm{diag}(l_i)^{-1}O_i
$$

5. Write the completed $O_i$ and statistics $l_i,m_i$ (for possible recomputation during backpropagation) back from SRAM to HBM in one operation.

## III. Recomputation

During backpropagation, Flash Attention does not store the enormous attention weight matrix $P$. Instead, it recalculates it when needed, avoiding GPU-memory exhaustion from $N \times N$ space complexity.

### 1. Why Does Backpropagation Need $P$?

To understand backpropagation clearly, first list the three key steps of the forward pass:

1. Calculate the scores:

$$
S = QK^T
$$

2. Calculate the probabilities (Softmax):

$$
P = \mathrm{softmax}(S)
$$

3. Calculate the output:

$$
O = PV
$$

Suppose the final loss function is $\mathcal{L}$ and the objective is to obtain $\frac{\partial \mathcal{L}}{\partial Q}$, $\frac{\partial \mathcal{L}}{\partial K}$, and $\frac{\partial \mathcal{L}}{\partial V}$. Also suppose we already have:

$$
dO = \frac{\partial \mathcal{L}}{\partial O}
$$

First calculate $dV$:

$$
dV = \frac{\partial \mathcal{L}}{\partial V} = P^T \cdot dO
$$

This step already requires $P$.

Next, calculate the gradient propagated from $O = PV$ back to $P$:

$$
dP = dO \cdot V^T
$$

This step uses $V$ and does not yet require $P$. However, we must next deal with $dP/dS$. Note that Softmax is not linear: $dP/dS$ cannot be directly expressed using constant parameters in the same way that $V^T$ expresses $dP/dO$. Instead, the derivative of Softmax necessarily uses the values of $P$ (or $S$).

The following expression provides an intuitive illustration of how the derivative of this kind of normalization function depends on the output value itself:

$$
\sigma'(x) = \sigma(x)\cdot(1-\sigma(x))
$$

Strictly speaking, the Jacobian matrix of vector Softmax satisfies:

$$
\frac{\partial P_i}{\partial S_j} = P_i(\delta_{ij}-P_j)
$$

In matrix form, $dS$ in Softmax backpropagation is calculated as follows ($\odot$ denotes elementwise multiplication):

$$
dS = P \odot \left(dP - (dP \odot P)1\right)
$$

Here, $1$ denotes an all-ones vector representing row-wise summation. Looking closely at this formula, we see $P$ throughout it. Intuitively, Softmax is very flat at both ends, where probabilities are close to 0 or 1, and the gradients are small; in the middle region, the gradients are large. To calculate the gradient $dS$, we must know whether the current probability $P$ lies in a flat or steep region.

Continuing backpropagation to $Q$ and $K$, $S = QK^T$ gives:

$$
dQ = dS K,\quad dK = dS^T Q
$$

If scaled attention $S = QK^T/\sqrt{d}$ is used, the expressions above must also be multiplied by the corresponding scale factor $1/\sqrt{d}$.

The conclusion is that, without knowing $P$, gradients cannot pass through the Softmax layer to $Q$ and $K$.

### 2. The Recomputation Method

For the two reasons above, $P$ is absolutely indispensable when calculating gradients during backpropagation.

Standard attention chooses to store it.

- Once $P$ is calculated during the forward pass, it is stored in HBM for backpropagation, despite being as large as $N \times N$. This is the source of GPU-memory exhaustion.

Flash Attention chooses to calculate it on demand.

- Since $P$ must be used but we do not want to store it, at the moment when backpropagation needs $P$, the forward computation is repeated using the stored $Q,K$ and statistics in SRAM:

$$
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)
$$

This produces a small block of $P$, which is discarded immediately after use.

Therefore, Flash Attention stores the lower-order inputs and statistics required for backpropagation, such as $Q,K,V,O,l,m$, rather than the complete $P$. This replaces storage of the original $O(N^2)$ intermediate matrix with blockwise recomputation during backpropagation.

### 3. Why Store $Q,K,V$ but Not $P$?

Simply put, we do not focus on $Q,K,V$ because they are too "small," whereas $P$ (and $S$) is too "large."

Here, "large" and "small" refer to different rates of growth in GPU-memory consumption as sequence length $N$ increases:

- The size of $Q,K,V$ grows linearly, with complexity $O(Nd)$.
- The size of $P$ grows quadratically, with complexity $O(N^2)$.

When $N$ becomes very large, as in long-text processing, the quadratic term is the core cause of GPU-memory exhaustion.

Specifically:

- For $Q,K,V$: because they are small, with complexity $O(Nd)$, and recomputing them is also costly because it requires recalculating linear layers from lower-level inputs, Flash Attention caches $Q,K,V$ in HBM.
- For $P$: because it is enormous, with complexity $O(N^2)$, storing it causes out-of-memory (OOM) errors and incurs enormous I/O costs. Flash Attention therefore chooses not to store it and instead recalculates it using $Q,K,V$.

This is the core of Flash Attention's GPU-memory savings: retain necessary state of linear size, discard intermediate attention matrices of quadratic size, and recompute them blockwise when needed during backpropagation.

## References

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). NeurIPS 2022.
- Dao, T. (2023). [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). arXiv:2307.08691.
- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608). arXiv:2407.08608.
