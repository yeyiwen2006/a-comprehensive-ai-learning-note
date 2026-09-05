---
title: "19.4 Multi-Query and Grouped-Query Attention"
chapter_title: "Engineering Optimizations for Attention"
section_id: "19-04"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.4 Multi-Query与Grouped-Query Attention.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.4 Multi-Query and Grouped-Query Attention

## I. The Basic Idea of MQA

In MHA, GPU-memory consumption becomes enormous as sequences grow longer. Memory bandwidth becomes the core constraint. Inference primarily involves matrix–vector multiplication (GEMV), a typical memory-bandwidth-bound operation. The GPU spends far more time "moving KV cache data" than "computing." The core idea of multi-query attention (MQA) is extremely straightforward: all query heads share the same set of key and value heads.

MHA (multi-head): H query heads, H key heads, H value heads.

MQA (multi-query): H query heads, 1 key head, 1 value head.

**Parameter Count Comparison**

- **MHA**: the total size of the parameter matrices is approximately $3d_{model}^2$, because the $Q$, $K$, and $V$ projections each account for one copy.
- **MQA**: the parameter matrix $W_Q$ remains $d_{model} \times d_{model}$, but both $W_K$ and $W_V$ shrink to $d_{model} \times d_k$.
- Because MQA retains only one set of key/value heads, the parameter counts of $W_K,W_V$ are reduced by a factor of $H$ relative to MHA.

When calculating attention scores for head $i$, MQA retains a multi-head query projection, while key/value projections use only one head shared by all query heads.

- **Query projection (remaining multi-head)**:

$$
Q_i = XW_Q^i, \quad Q_i \in \mathbb{R}^{B \times L \times d_k}
$$

- **Key/value projection (single-head)**:

$$
K = XW_K, \quad V = XW_V, \quad K,V \in \mathbb{R}^{B \times L \times d_k}
$$

Here, $K$ and $V$ have no subscript $i$, indicating that all heads share this set of keys and values.

- **Broadcasting and computation**: for row-wise dot-product attention computation, $K$ and $V$ must be logically broadcast along the head dimension so that their shapes match $Q_i$.

$$
\mathrm{Attention}_i(Q_i,K,V)=\mathrm{softmax}\left(\frac{Q_iK^T}{\sqrt{d_k}}\right)V
$$

Finally, concatenate the outputs of all heads and pass them through the $W_O$ output layer.

**Why Is MQA Faster?**

- **KV cache reduced by a factor of $H$**: because all heads share $K,V$, the amount of KV cache data to store becomes $1/H$ of the original; for example, when $H=8$, GPU-memory usage is approximately $1/8$ of the original.
- **Reduced memory-bandwidth pressure**: during inference, the amount of data the GPU must move from GPU memory decreases substantially, alleviating the memory-wall problem.
- **Higher TPS**: after GPU-memory consumption and memory-bandwidth pressure decrease, token generation speed (TPS) usually increases.

## II. Grouped-Query Attention (GQA)

MQA improves speed but can readily affect generation quality. To balance MHA's high quality and MQA's high speed, modern models introduce GQA, grouping query heads so that each group shares one key/value head.

For example, 8 query heads are divided into 4 groups, with every 2 query heads sharing 1 KV head.

The KV cache size is 1/2 that of MHA and twice that of MQA.

Effect: speed close to MQA, quality close to MHA.

## References

- Shazeer, N. (2019). [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). arXiv:1911.02150.
- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., & Sanghai, S. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023.
