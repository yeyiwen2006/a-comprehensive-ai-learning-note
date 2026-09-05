---
title: "19.10 Gated Attention"
chapter_title: "Engineering Optimizations for Attention"
section_id: "19-10"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.10 门控注意力（Gated Attention）.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.10 Gated Attention

> This article is a paper-reading note. Its content represents the corresponding paper's method or the author's understanding and should not be treated directly as field consensus or engineering best practice.

Unlike the preceding improvements, which focus on efficiency, gated attention proposed by the Qwen team focuses more on improving performance.

## I. Problems Addressed

1. Low-rank bottleneck

In standard multi-head attention (MHA), for any head, the output Y weights the tokens' value matrix according to their attention scores:

$$
Y=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The attention matrix is:

$$
A=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

In other words, standard attention can be written as $Y=AV$. Here, V is formed by concatenating multiple heads. For each head, the output matrix's rank is constrained by its dimensions and can only have $d_{head}$ dimensions. When discussing DeepSeek's innovations, we noted that the vectors of different heads are highly correlated, so they still lie in a low-rank space in practice. This may cause V to lose information in some dimensions of the high-dimensional data distribution. Both the attention layer after Softmax and the output layer are linear. If V fails to contain information in certain dimensions, it cannot recover that information through nonlinear "fitting."

2. Unreasonable weight allocation

In actual operation, many earlier models often assign a large proportion of attention scores to the first token in a sentence (such as the start token) when all token keys have low relevance to the query, even if that token has no actual semantic meaning. This may occur because all token attention weights are forced to sum to 1, making the first token a "wastebasket" to maintain numerical stability.

## II. Solution

Introduce a gate after the attention-layer output:

$$
Y_{\mathrm{gated}}=(AV)\odot\sigma(XW_{\mathrm{gate}})
$$

Let the gating matrix be:

$$
G=\sigma(XW_{\mathrm{gate}})
$$

Gated attention can then be written as $Y_{\mathrm{gated}}=(AV)\odot G$. Here, $\odot$ denotes elementwise multiplication, and every element of $G$ is obtained from input $X$ through a linear transformation and a sigmoid function.

According to a corollary of the Schur product theorem, for a matrix $A$ of rank $r_1$ and a matrix $B$ of rank $r_2$, the rank of their elementwise product $A\odot B$ is bounded above by:

$$
\mathrm{Rank}(A\odot B)\le \mathrm{Rank}(A)\times \mathrm{Rank}(B)
$$

Note that this is elementwise multiplication, not ordinary matrix multiplication. The rank bound becomes the product of the ranks of the two matrices, greatly relaxing the constraint. This can also be understood geometrically:

The output of standard attention is essentially a convex combination of value vectors. Because the attention weights after Softmax are nonnegative and sum to 1, the output vector must lie inside the convex hull of all value vectors, so the model cannot generate a point outside these vectors.

Gated attention introduces a gating matrix $G\in(0,1)$ after this output. The gate independently scales each vector dimension, equivalent to nonuniformly stretching or compressing a point inside the convex hull. When gate values in certain dimensions approach 0, those dimensions are suppressed; when gating strengths differ across dimensions, the output point is no longer merely a convex combination of the original value vectors. This breaks the geometric restriction of "convex combinations" and enriches the output space.

Experiments show that the gating matrix is often sparse, which also addresses the unreasonable weight allocation described above.

## References

- Qiu, Z., Wang, Z., Zheng, B., et al. (2025). [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708). arXiv:2505.06708.
