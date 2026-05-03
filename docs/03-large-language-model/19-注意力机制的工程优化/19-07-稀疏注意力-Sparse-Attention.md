---
title: "19.7 稀疏注意力（Sparse Attention）"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.7 稀疏注意力（Sparse Attention）.docx"
status: "auto-converted"
ocr: "auto-generated, needs human review"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.7 稀疏注意力（Sparse Attention）

> 本文由本地 Word 原稿自动转换而来。图片中的文字由 OCR 自动识别，可能存在识别错误，欢迎提交 Issue 修正。

## 一、核心思想：认为并非所有 token 对之间都需要进行交互。通过引入结构性稀疏模式，只计算注意力矩阵中一部分重要的元素，将复杂度从O(n^2) 降低到 O(n*sqrt(n))或O(nlogn)。

## 二、典型代表

Blockwise Attention：将序列分块，只在块内或特定的块之间计算注意力。

Sliding Window Attention（滑动窗口注意力）：每个 token 只关注其前后一定窗口大小内的邻居 token。这在类似BERT这样的编码器中很常见。

Dilated Attention：类似于空洞卷积，在滑动窗口的基础上进行跳跃，以扩大感受野。

Global + Local Attention：指定少数 token（如 <s>）拥有全局注意力，可以关注所有 token，而其他 token 只进行局部注意力。

应用：Longformer、BigBird 等模型就采用了这种思想，能够处理极长文档。

## 三、DSA（Dynamic Sparse Attention，DeepSeek-V3.2，2025）

1.核心思想：“先粗筛，后细算”

> [图片 1：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

目标： 在不计算所有 Query-Key 点积的情况下， 快速找到高价值的 Keyo 输入： Query 侧的索引信号： 从嗨衍生出的 qtjo · Key 侧的索引信号：。 权重 / 亻扁置项： 0。 处理流程： 部分 RoPE (Partially apply RoPE): 为了降低计算量， 索引器不需要全精度的位置纟扁码， 1 ． 只对索引向量应用部分 RoPE0 Lightning lndexer （闪电索引器）： 这是一个轻量级的计算模块。 它接收粗粒度的 Query 和 2 ． Key 索引向量， 快速计算粗略的相关性分数。 3 ． Top-k Selector (Top-k 选择器）： 基于索引器计算出的分数， 动态选择出当前最关注的个 Key-Value 块。

2.推理工作流

> [图片 2：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

Ste p 1： 投影降维 (Down-Projection) 原理： 模型不直接使用高维的或完整的进行检索， 而是通过一个轻量级的线性层彬 7 将输入投影到一个极低维度的空间。 数学表示 · 7 一 · （Query 侧索引向量） 7 KV (Key 侧索引向量） 关键点： 这里的索引向量维度 “ 远小于正常的 Head 维度 dheado 例如， dhead 可能是 128， 而 “ 可能只有 32 或 16。 这就极大减少了计算量。 Ste p 2： 块级划分 (BIock-wise Segmentation) 原理： " 粗粒度 " 不仅指向量维度低， 还指时间步的粒度。 DSA 通常不会对每一个历史 Token 逐个计算分数， 而是将 KV Cache 划分为多个固定大小的 Block （块） （例如每 64 个 Token 为一块）。 · 操作： 索引器为每一个 Block 计算一个代表性的 Key 索引向量 （通常是该 Block 内所有的均值或特定位置的采样）。

> [图片 3：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

Ste p 1： 投影降维 (Down-Projection) 原理： 模型不直接使用高维的或完整的进行检索， 而是通过一个轻量级的线性层彬 7 将输入投影到一个极低维度的空间。 数学表示 · 7 一 · （Query 侧索引向量） 7 KV (Key 侧索引向量） 关键点： 这里的索引向量维度 “ 远小于正常的 Head 维度 dheado 例如， dhead 可能是 128， 而 “ 可能只有 32 或 16。 这就极大减少了计算量。 Ste p 2： 块级划分 (BIock-wise Segmentation) 原理： " 粗粒度 " 不仅指向量维度低， 还指时间步的粒度。 DSA 通常不会对每一个历史 Token 逐个计算分数， 而是将 KV Cache 划分为多个固定大小的 Block （块） （例如每 64 个 Token 为一块）。 · 操作： 索引器为每一个 Block 计算一个代表性的 Key 索引向量 （通常是该 Block 内所有的均值或特定位置的采样）。

> [图片 4：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

Ste p 3： 快速打分 (Lightweight Scoring) 原理： 使用点积计算相关性分数。 数学表示： block block 注意图中提到的 "partially apply RoPE： 为了进一步节省算力， 同时保留位置信息， 索引器只对向量的一小部分维度应用旋转位置编码， 或者使用简化版的位置编码。 Ste p 4： Top-k 筛选 (Gating) 根据 Sblock 的大小， 选出分数最高的个 Block0 结果： 后续的 " 重型 " 注意力计算 ℃ ore Attention) 只针对这个被选中的 Block 进行加载和计算， 其余 90 ％ + 的无关信息被直接忽略。

> [图片 5：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

为什么它能 “ 快 "？ 计算量级差异： 假设原计算量是 0 · “ 刁。 索引器的计算量是 0 · dindex）， 其中 B 是块大小。 由于 B > 1 且 “ 《 dhead， 索引器的开销几乎可以忽略不计。

3.训练方法

（1）Lightning Indexer的训练目标

由于Top-K操作本身是不可导的（即无法直接通过Top-K选择反向传播主模型的预测误差来更新索引器），DeepSeek为Lightning Indexer设计了独立的监督信号。

Lightning Indexer被训练来预测对于每个token的Query和每个token的Key之间的注意力权重，训练目标是模仿原始全注意力（Dense Attention）的权重分布（准确说是所有注意力头的权重的平均值的分布）。

损失函数是n个KL散度的和：对于任意第i个token，它对第j个token的注意力分数（j=1,2,...,n）构成概率分布p(j|i)，我们计算p_Indexer(j|i)相对于p_Dense(j|i)的KL散度，再对i=1,2,...,n累加。

（2）整体训练步骤

第一步：在全注意力（Dense Attention）下，训练主模型。

第二步：冻结主模型，保持全注意力（Dense Attention）开启，初始化训练Lightning Indexer使之与主模型的注意力分布对齐。

第三步：开启稀疏化（Top-K Selection），同步用各自的损失函数训练主模型和Lightning Indexer。

## 参考文献与引用线索

> 本节由脚本自动检索正文中的引用线索，可能不完整；未能确定来源的位置会在下方标为待补引用。

### 待补引用或版权检查

- [待补引用] 本文含 Word 内嵌图片；开源版未上传图片。若图片来自教材、论文或技术报告，建议人工确认授权、补充来源或重画。
- [待补引用] 未自动检索到明确参考文献线索，建议人工补充可追溯来源。
