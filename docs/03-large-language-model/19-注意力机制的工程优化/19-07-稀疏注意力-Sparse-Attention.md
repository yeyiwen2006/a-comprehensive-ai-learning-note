---
title: "19.7 稀疏注意力（Sparse Attention）"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.7 稀疏注意力（Sparse Attention）.docx"
status: "synced-from-docx"
ocr: "manual reconstruction completed; text and formula screenshots transcribed as Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.7 稀疏注意力（Sparse Attention）

## 一、核心思想

稀疏注意力的核心思想是：并非所有 token 对之间都需要进行交互。通过引入结构性稀疏模式，只计算注意力矩阵中一部分重要元素，可以把全注意力的 $O(n^2)$ 复杂度降低到 $O(n\sqrt{n})$、$O(n\log n)$ 或近似线性的级别。

## 二、典型代表

Blockwise Attention：将序列分块，只在块内或特定的块之间计算注意力。

Sliding Window Attention（滑动窗口注意力）：每个 token 只关注其前后一定窗口大小内的邻居 token。这在类似 BERT 这样的编码器中很常见。

Dilated Attention：类似于空洞卷积，在滑动窗口的基础上进行跳跃，以扩大感受野。

Global + Local Attention：指定少数 token（如 `<s>`）拥有全局注意力，可以关注所有 token，而其他 token 只进行局部注意力。

应用：Longformer、BigBird 等模型就采用了这种思想，能够处理极长文档。

## 三、DSA（DeepSeek Sparse Attention，DeepSeek-V3.2，2025）

### 1. 核心思想：“先粗筛，后细算”

DSA 的目标是在不计算所有 Query-Key 点积的情况下，快速找到高价值的 Key。

输入包括：

- Query 侧索引信号：从 $c_t^Q$ 衍生出的 $q^I_{t,j}$。
- Key 侧索引信号：$k_t^I$。
- 权重或偏置项：$w^I_{t,j}$。

处理流程包括：

1. 部分 RoPE（Partially apply RoPE）：为了降低计算量，索引器不需要全精度的位置编码，只对索引向量应用部分 RoPE。
2. Lightning Indexer（闪电索引器）：这是一个轻量级计算模块。它接收粗粒度的 Query 和 Key 索引向量，快速计算粗略的相关性分数。
3. Top-k Selector（Top-k 选择器）：基于索引器计算出的分数，动态选出当前 Query 最关注的 $k$ 个 Key-Value 块。

### 2. 推理工作流

Step 1：投影降维（Down-Projection）。

模型不直接使用高维的 $h_t$ 或完整的 $c_t^Q$ 进行检索，而是通过一个轻量级线性层 $W^I$ 将输入投影到一个低维空间：

$$
q^I_{t,j} = W^I_Q \cdot c^Q_t
$$

$$
k^I_t = W^I_K \cdot c^{KV}_t
$$

其中，$q^I_{t,j}$ 是 Query 侧索引向量，$k^I_t$ 是 Key 侧索引向量。这里的索引向量维度 $d_{index}$ 远小于正常的 head 维度 $d_{head}$。例如，$d_{head}$ 可能是 128，而 $d_{index}$ 可能只有 32 或 16，这会显著减少索引阶段的计算量。

Step 2：块级划分（Block-wise Segmentation）。

这里的“粗粒度”不仅指向量维度低，也指时间步的粒度。DSA 通常不会对每一个历史 token 逐个计算分数，而是将 KV Cache 划分为多个固定大小的 Block（块），例如每 64 个 token 为一块。

索引器会为每一个 Block 计算一个代表性的 Key 索引向量，通常可以取该 Block 内所有 $k^I$ 的均值，也可以对特定位置采样。

Step 3：快速打分（Lightweight Scoring）。

使用点积计算 Query 索引向量与 Block 索引向量之间的相关性分数：

$$
S_{block} = (q^I_{t,j})^T \cdot k^I_{block}
$$

图中提到的 “partially apply RoPE” 指的是：为了进一步节省算力，同时保留位置信息，索引器只对向量的一小部分维度应用旋转位置编码，或者使用简化版的位置编码。

Step 4：Top-k 筛选（Gating）。

根据 $S_{block}$ 的大小，选出分数最高的 $k$ 个 Block。后续真正的“重型”注意力计算（Core Attention）只针对这 $k$ 个被选中的 Block 进行加载和计算，其余大量无关信息直接忽略。

之所以这种方法更快，是因为原始全注意力的计算量大致为：

$$
O(L \cdot d_{head})
$$

而索引器的计算量大致为：

$$
O\left(\frac{L}{B} \cdot d_{index}\right)
$$

其中，$B$ 是块大小。由于 $B > 1$ 且 $d_{index} \ll d_{head}$，索引器开销通常远小于完整注意力计算。

### 3. 训练方法

（1）Lightning Indexer 的训练目标。

由于 Top-K 操作本身是不可导的，也就是无法直接通过 Top-K 选择反向传播主模型的预测误差来更新索引器，DeepSeek 为 Lightning Indexer 设计了独立的监督信号。

Lightning Indexer 被训练来预测每个 token 的 Query 与每个 token 的 Key 之间的注意力权重。训练目标是模仿原始全注意力（Dense Attention）的权重分布，准确说是所有注意力头的权重平均分布。

损失函数可以理解为 $n$ 个 KL 散度之和。对于第 $i$ 个 token，它对第 $j$ 个 token 的注意力分数（$j = 1, 2, \ldots, n$）构成概率分布 $p(j \mid i)$。再计算索引器分布 $p_{Indexer}(j \mid i)$ 与稠密注意力分布 $p_{Dense}(j \mid i)$ 之间的 KL 散度，并对 $i = 1, 2, \ldots, n$ 累加：

$$
\mathcal{L}_{indexer}
= \sum_{i=1}^{n} D_{KL}\left(p_{Dense}(\cdot \mid i) \,\|\, p_{Indexer}(\cdot \mid i)\right)
$$

（2）整体训练步骤。

第一步：在全注意力（Dense Attention）下，训练主模型。

第二步：冻结主模型，保持全注意力开启，初始化训练 Lightning Indexer，使之与主模型的注意力分布对齐。

第三步：开启稀疏化（Top-K Selection），同步用各自的损失函数训练主模型和 Lightning Indexer。

## 参考文献

- Qiu, J., Ma, H., Levy, O., Yih, S. W., Wang, S., & Tang, J. (2019). [Blockwise Self-Attention for Long Document Understanding](https://arxiv.org/abs/1911.02972). arXiv:1911.02972.
- Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509). arXiv:1904.10509.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). arXiv:2004.05150.
- Zaheer, M., Guruganesh, G., Dubey, K. A., et al. (2020). [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062). NeurIPS 2020.
- Ding, J., Ma, S., Dong, L., et al. (2023). [LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486). arXiv:2307.02486.
- DeepSeek-AI. (2025). [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556). arXiv:2512.02556.
