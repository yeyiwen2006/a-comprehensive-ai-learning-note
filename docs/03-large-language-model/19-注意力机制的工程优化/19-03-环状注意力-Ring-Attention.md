---
title: "19.3 环状注意力（Ring Attention）"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.3 环状注意力（Ring Attention）.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.3 环状注意力（Ring Attention）


Ring Attention（环状注意力）是分布式系统（多GPU/TPU）上的Flash Attention延伸，Google Gemini等模型均运用了此技术。

FlashAttention 解决的是单张显卡内，如何通过分块计算（Tiling）把超长序列塞进有限的 SRAM 中；Ring Attention 解决的是多张显卡间，如何通过分块轮转，把超长序列（比如100万+ Token）塞进集群的显存中，并打破单卡显存的物理上限。

假设你想训练一个上下文长度为1000万的模型。注意力机制要求Q必须和所有的K, V进行交互。即使使用了FlashAttention，单张H100也存不下这么长的KV Cache和中间激活值。这时候需要序列并行（Sequence Parallelism），把长序列切分到N张显卡上。

Ring Attention做法：每张显卡存一个Query块（Q矩阵的若干行，也就是若干个token的Query），让K, V数据块（每个数据块包含矩阵若干行）在显卡之间像“回转寿司”一样，通过高速互连网络流动，每张卡只处理流经它的那一小块数据，算完就传给下一张卡，绝不囤积数据。

这个过程会在环上的 $P$ 个节点之间执行 $P$ 轮：

1. **Inner Loop（计算）**：
   - GPU $i$ 使用本地的 $Q_i$ 和当前持有的 Key/Value 块（记为 $K_{\mathrm{curr}}, V_{\mathrm{curr}}$）计算注意力分数和局部输出。
   - 计算使用 FlashAttention 的 blockwise 逻辑，维护局部的 softmax 归一化因子和 log-sum-exp 统计量。

2. **Communication（通信）**：
   - 在计算的同时，GPU $i$ 将当前的 $K_{\mathrm{curr}}, V_{\mathrm{curr}}$ 发送给下一张卡（GPU $i+1$）。
   - GPU $i$ 同时从上一张卡（GPU $i-1$）接收新的 Key/Value 块。

3. **Overlap（计算与通信重叠）**：
   - Ring Attention 的精髓在于矩阵乘法计算通常比传输 Key/Value 块更耗时，因此二者可以并行推进。
   - 当系统计算第 $t$ 块数据时，网络已经在传输第 $t+1$ 块数据。
   - 如果计算时间大于传输时间，通信延迟就被完全隐藏，整体上接近零额外通信开销。

其中计算依赖于Softmax的分块计算性质：

Ring Attention 之所以能成立，依赖于 softmax 的分块计算性质，这也是 FlashAttention 的基础。标准 softmax 需要全局归一化：

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

但是，可以把序列切成两个块 $A$ 和 $B$，分别计算局部统计量，再把它们合并成全局统计量。

先计算块 $A$ 的局部最大值和局部指数和：

$$
m_A = \max_{j \in A} x_j,\qquad
l_A = \sum_{j \in A} e^{x_j - m_A}
$$

再计算块 $B$ 的局部最大值和局部指数和：

$$
m_B = \max_{j \in B} x_j,\qquad
l_B = \sum_{j \in B} e^{x_j - m_B}
$$

最后用简单的数学变换合并，更新全局最大值 $m_{\mathrm{global}}$ 和全局归一化因子 $l_{\mathrm{global}}$：

$$
m_{\mathrm{global}} = \max(m_A, m_B)
$$

$$
l_{\mathrm{global}} =
e^{m_A - m_{\mathrm{global}}} l_A
+ e^{m_B - m_{\mathrm{global}}} l_B
$$

因此，Ring Attention 每次只需要在显存里放一个 $K, V$ block，算完就更新统计量，然后把该 block 挪走。

计算的具体步骤如下：

对第 $i$ 张显卡而言，当前轮只使用本地 query 块 $Q_i$ 和当前持有的 $K_{\mathrm{curr}}, V_{\mathrm{curr}}$。以下统计量都按 query 行分别计算，矩阵形式中相应的最大值和求和会按行广播。

计算局部分数：

$$
S = \frac{Q_i K_{\mathrm{curr}}^T}{\sqrt{d}}
$$

计算当前块的行最大值：

$$
m_{\mathrm{block}} = \max(S)
$$

计算当前块的局部指数和：

$$
l_{\mathrm{block}} = \sum \exp(S - m_{\mathrm{block}})
$$

计算非归一化的注意力输出：

$$
\tilde O_{\mathrm{block}} = \exp(S - m_{\mathrm{block}}) V_{\mathrm{curr}}
$$

第i张显卡的输出O是一个和Q_i行数相同的矩阵（如果每张显卡只存一个Query向量，则O就是一个向量），是对于每个Query，根据目前已经经过该显卡的K、V得到的对V进行注意力加权的结果。每一轮计算后，更新统计量和输出O：

假设上一轮已经维护了 $m_{\mathrm{prev}}$、$l_{\mathrm{prev}}$ 和 $O_{\mathrm{prev}}$，当前块给出了 $m_{\mathrm{block}}$、$l_{\mathrm{block}}$ 和 $\tilde O_{\mathrm{block}}$。online softmax 的合并方式如下。

更新全局最大值：

$$
m_{\mathrm{new}} = \max(m_{\mathrm{prev}}, m_{\mathrm{block}})
$$

更新归一化因子：

$$
l_{\mathrm{new}} =
e^{m_{\mathrm{prev}} - m_{\mathrm{new}}} l_{\mathrm{prev}}
+ e^{m_{\mathrm{block}} - m_{\mathrm{new}}} l_{\mathrm{block}}
$$

更新输出 $O$：

$$
O_{\mathrm{new}} =
\frac{
l_{\mathrm{prev}} e^{m_{\mathrm{prev}} - m_{\mathrm{new}}} O_{\mathrm{prev}}
+ e^{m_{\mathrm{block}} - m_{\mathrm{new}}} \tilde O_{\mathrm{block}}
}{
l_{\mathrm{new}}
}
$$

这样，每张卡不需要保存完整的 $K, V$，只要在每一轮合并局部统计量和局部输出，就能得到等价于全局 softmax 的注意力结果。

## 参考文献

- Liu, H., Zaharia, M., & Abbeel, P. (2023). [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889). arXiv:2310.01889.
