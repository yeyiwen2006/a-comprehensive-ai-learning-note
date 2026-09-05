---
title: "19.1 KV Cache"
chapter_title: "Engineering Optimizations for Attention"
section_id: "19-01"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.1 键值缓存（KV Cache）.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.1 KV Cache

## I. The Core Principle of KV Cache

In full attention, generating each token has time and space complexity O(n^2) (using a KV cache during inference reduces the time complexity for generating a single token to O(n), gives O(n^2) time complexity for generating a sequence, and O(n) memory complexity). These costs are the main obstacles to processing long sequences (long contexts). In particular, during inference (not training), generating each token incurs an O(L^2) computational cost. However, much of the computation of the K and V matrices from the input sequence is repeated.

To address the computational time complexity during inference, LLMs commonly use a KV cache:

1. When calculating Q, K, and V from the input

In traditional attention:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

With a KV cache, when generating token t+1, we only need token t as Q. For K and V, rows 1 through t-1 (the K and V values of tokens 1 through t-1) have already been calculated during previous generation steps and can be stored in memory for direct reuse. Only the K and V values of token t need to be calculated:

$$
q_t = x_tW_Q,\quad k_t = x_tW_K,\quad v_t = x_tW_V
$$

Here, k_t and v_t are appended below the previously calculated K and V matrices in memory for subsequent use. This reduces the computational complexity from O(L) to O(1).

2. When calculating attention weights

In traditional attention:

$$
\mathrm{Scores} = Q \cdot K^T
$$

Description of the operation:

- The dimensions of $Q$ are $t \times d_k$.
- The dimensions of $K$ are $t \times d_k$, so those of $K^T$ are $d_k \times t$.
- This is a standard matrix multiplication, yielding an attention score matrix of dimensions $t \times t$.

With a KV cache:

$$
\mathrm{Scores} = q_t \cdot (K_{\mathrm{cache}}^{(t)})^T
$$

Description of the operation:

- The dimensions of $q_t$ are $1 \times d_k$.
- The dimensions of $K_{\mathrm{cache}}^{(t)}$ are $t \times d_k$ ($d_k \times t$ after transposition).
- This is a "vector–matrix" multiplication, yielding attention scores of dimensions $1 \times t$ for the current step.

The computational complexity is therefore reduced from O(L^2) to O(L).

3. When calculating the output by weighting

In traditional attention:

$$
Y = \mathrm{Softmax}(\mathrm{Scores}) \cdot V
$$

Description of the operation:

- The attention matrix has dimensions $t \times t$.
- The $V$ matrix has dimensions $t \times d_v$.
- The output $Y$ has dimensions $t \times d_v$.

With a KV cache:

$$
y_t = \alpha_t \cdot V_{\mathrm{cache}}^{(t)}
$$

Description of the operation:

- The weights $\alpha_t$ have dimensions $1 \times t$.
- $V_{\mathrm{cache}}^{(t)}$ has dimensions $t \times d_v$.
- This is a "vector–matrix" multiplication, and the output $y_t$ has dimensions $1 \times d_v$.

## II. Why Is KV Cache Not Used during Training?

A KV cache is designed to address the inefficiency of "serial computation" when generating one token at a time during inference. Training, however, is naturally parallel: we already know the complete answer, and teacher forcing and masking allow us to calculate the losses at all positions in one computation. There is therefore no need to "cache the previous step's result for the next step." Forcibly using a KV cache during training would instead turn efficient parallel training into inefficient serial training, making training hundreds or thousands of times slower.

As discussed earlier, during pretraining or fine-tuning, the steps are completed simultaneously in a single matrix operation.

We obtain Q, K, and V for all tokens at once, then calculate attention scores and apply a mask. The mask is an "upper-triangular matrix" that acts after attention score calculation and before Softmax. It forces the attention scores for all future positions (i,j) (i>j), that is, the dot products of q_i and k_j, to a negative number with an extremely large absolute value (for example, -1e9).

There is no situation in this process where "step 1 finishes and stores its result for step 2," because steps 1 and 2 occur simultaneously.

[Special case: during PPO training in RLHF, the procedure has two steps. The rollout generation stage lets the model generate responses to prompts; this is essentially inference, so a KV cache is used. The update stage uses the generated responses and rewards to calculate gradients and update the model, returning to parallel operation without a KV cache.]

## III. Prefill

Before generation starts, we let the model read the entire prompt in parallel in one pass, calculate the initial Q, K, and V matrices using Q = X*W_Q, K = X*W_K, and V = X*W_V, and store the initial KV cache. Decoding then begins, generating tokens autoregressively.

Prefill is computation-intensive, whereas decoding is memory-intensive. Some architectures assign the two stages to GPUs with high computational power and high GPU-memory bandwidth, respectively.

## IV. System Hardware Management: Paged Attention (vLLM)

A KV cache requires contiguous space in GPU memory, causing considerable waste from memory fragmentation (similar to fragmentation in operating-system memory). Borrowing the paging technique from operating-system virtual memory, it stores the KV cache in blocks in noncontiguous physical GPU memory and indexes them through a page table. This greatly improves GPU-memory utilization, allowing a larger batch size with the same memory and thereby improving inference throughput.

## References

- Kwon, W., Li, Z., Zhuang, S., et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.
