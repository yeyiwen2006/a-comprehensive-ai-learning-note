---
title: "19.4 Multi-Query与Grouped-Query Attention"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.4 Multi-Query与Grouped-Query Attention.docx"
status: "auto-converted"
ocr: "auto-generated, needs human review"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.4 Multi-Query与Grouped-Query Attention

> 本文由本地 Word 原稿自动转换而来。图片中的文字由 OCR 自动识别，可能存在识别错误，欢迎提交 Issue 修正。

## 一、MQA的基本思想

在MHA中，随着序列变长，显存占用巨大。内存带宽（Memory Bandwidth）成为核心限制。推理过程主要是矩阵乘向量（GEMV），是典型的内存带宽受限操作。GPU 花在“搬运 KV Cache 数据”上的时间远多于“计算”的时间。MQA（Multi-Query Attention）的核心思想非常简单粗暴：所有的 Query 头之间共享同一组 Key 和 Value 头。

MHA（Multi-Head）：H个 Query Heads，H个Key Heads，H个 Value Heads

MQA（Multi-Query）：H个 Query Heads，1个 Key Head，1个 Value Head

> [图片 1：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

参数量对比 MHA: 参数矩阵总大小约为 3 × d2 （因为 Q,K, 各占一份）。 model MQA: 参数矩阵 WQ 大小仍为 dmodel × dmodel， 但 WK 和彬急剧缩小为 dmodel × 嘲。 彬五， Wv 的参数量减少了 H 倍。

计算第i个头的注意力分数时：

> [图片 2：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

M QA 中， 计算第个头的注意力分数时： Query 投影 （保持多头）： Key/VaIue 投影 （单头）： K 一 W_， V 一 XWv,： 这里 K 和没有下标 i, 所有头共用这一组。 B><L><dk K, V e IR 广播 (Broadcasting) 与计算： 为了进行点积注意力计算， 我们需要在逻辑上将和 V 沿着 Head 维度进行广播 （复制 ), 使其形状与 Q 匹配。 QKT Attentiont (Qi,，） = softmax 最终将所有头的输出拼接 ℃ oncat) 并通过 0 输出层。

> [图片 3：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

为什么 M QA 更快？ KVCache 减少丑倍： 由于所有头共享，， 我们需要存储的 KV Cache 数据量直接减少了 H 倍 （例如， 如果是 8 个头， 显存占用就降为原来的 1 / 8）。 降低内存带宽压力： 在推理时， GPU 需要从显存搬运的数据量大幅减少， 从而缓解了 Memory wall （内存墙） 问题， 显著提升了 Token 生成速度 (TPS)0

## 二、GQA（Grouped-Query Attention）

MQA提高了速度，却容易影响生成质量。为了平衡MHA的高质量和MQA的高速度，现代模型引入了GQA，将Query头分组，每组共享一个Key/Value 头。

如：8个Query头，分为4组，每组 2 个Query头共享1个KV头。

KV Cache大小是MHA的1/2，是MQA的2倍。

效果：速度接近 MQA，质量接近 MHA。

## 参考文献与引用线索

> 本节由脚本自动检索正文中的引用线索，可能不完整；未能确定来源的位置会在下方标为待补引用。

### 待补引用或版权检查

- [待补引用] 本文含 Word 内嵌图片；开源版未上传图片。若图片来自教材、论文或技术报告，建议人工确认授权、补充来源或重画。
- [待补引用] 未自动检索到明确参考文献线索，建议人工补充可追溯来源。
