---
title: "19.4 Multi-Query与Grouped-Query Attention"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.4 Multi-Query与Grouped-Query Attention.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.4 Multi-Query与Grouped-Query Attention


## 一、MQA的基本思想

在MHA中，随着序列变长，显存占用巨大。内存带宽（Memory Bandwidth）成为核心限制。推理过程主要是矩阵乘向量（GEMV），是典型的内存带宽受限操作。GPU 花在“搬运 KV Cache 数据”上的时间远多于“计算”的时间。MQA（Multi-Query Attention）的核心思想非常简单粗暴：所有的 Query 头之间共享同一组 Key 和 Value 头。

MHA（Multi-Head）：H个 Query Heads，H个Key Heads，H个 Value Heads

MQA（Multi-Query）：H个 Query Heads，1个 Key Head，1个 Value Head

**参数量对比**

- **MHA**：参数矩阵总大小约为 $3d_{model}^2$，因为 $Q$、$K$、$V$ 三个投影各占一份。
- **MQA**：参数矩阵 $W_Q$ 大小仍为 $d_{model} \times d_{model}$，但 $W_K$ 和 $W_V$ 都缩小为 $d_{model} \times d_k$。
- 因为 MQA 只保留 1 组 Key/Value 头，$W_K,W_V$ 的参数量相对 MHA 减少 $H$ 倍。

计算第 $i$ 个头的注意力分数时，MQA 保持 Query 投影为多头，而 Key/Value 只做单头投影并被所有 Query 头共享。

- **Query 投影（保持多头）**：

$$
Q_i = XW_Q^i, \quad Q_i \in \mathbb{R}^{B \times L \times d_k}
$$

- **Key/Value 投影（单头）**：

$$
K = XW_K, \quad V = XW_V, \quad K,V \in \mathbb{R}^{B \times L \times d_k}
$$

这里 $K$ 和 $V$ 没有下标 $i$，表示所有头共用这一组 Key/Value。

- **广播（Broadcasting）与计算**：为了逐行点积注意力计算，需要在逻辑上将 $K$ 和 $V$ 沿着 Head 维度广播，使其形状与 $Q_i$ 匹配。

$$
\mathrm{Attention}_i(Q_i,K,V)=\mathrm{softmax}\left(\frac{Q_iK^T}{\sqrt{d_k}}\right)V
$$

最后将所有头的输出拼接（Concat），并通过 $W_O$ 输出层。

**为什么 MQA 更快？**

- **KV Cache 减少 $H$ 倍**：由于所有头共享 $K,V$，需要存储的 KV Cache 数据量直接变为原来的 $1/H$；例如 $H=8$ 时，显存占用约为原来的 $1/8$。
- **降低内存带宽压力**：在推理时，GPU 需要从显存搬运的数据量大幅减少，从而缓解 Memory Wall（内存墙）问题。
- **提升 TPS**：显存占用和内存带宽压力下降后，Token 生成速度（TPS）通常会提升。

## 二、GQA（Grouped-Query Attention）

MQA提高了速度，却容易影响生成质量。为了平衡MHA的高质量和MQA的高速度，现代模型引入了GQA，将Query头分组，每组共享一个Key/Value 头。

如：8个Query头，分为4组，每组 2 个Query头共享1个KV头。

KV Cache大小是MHA的1/2，是MQA的2倍。

效果：速度接近 MQA，质量接近 MHA。

## 参考文献

- Shazeer, N. (2019). [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). arXiv:1911.02150.
- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., & Sanghai, S. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023.
