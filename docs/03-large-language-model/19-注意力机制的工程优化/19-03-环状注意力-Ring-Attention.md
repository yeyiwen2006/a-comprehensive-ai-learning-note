---
title: "19.3 环状注意力（Ring Attention）"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.3 环状注意力（Ring Attention）.docx"
status: "auto-converted"
ocr: "auto-generated, needs human review"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.3 环状注意力（Ring Attention）

> 本文由本地 Word 原稿自动转换而来。图片中的文字由 OCR 自动识别，可能存在识别错误，欢迎提交 Issue 修正。

Ring Attention（环状注意力）是分布式系统（多GPU/TPU）上的Flash Attention延伸，Google Gemini等模型均运用了此技术。

FlashAttention 解决的是单张显卡内，如何通过分块计算（Tiling）把超长序列塞进有限的 SRAM 中；Ring Attention 解决的是多张显卡间，如何通过分块轮转，把超长序列（比如100万+ Token）塞进集群的显存中，并打破单卡显存的物理上限。

假设你想训练一个上下文长度为1000万的模型。注意力机制要求Q必须和所有的K, V进行交互。即使使用了FlashAttention，单张H100也存不下这么长的KV Cache和中间激活值。这时候需要序列并行（Sequence Parallelism），把长序列切分到N张显卡上。

Ring Attention做法：每张显卡存一个Query块（Q矩阵的若干行，也就是若干个token的Query），让K, V数据块（每个数据块包含矩阵若干行）在显卡之间像“回转寿司”一样，通过高速互连网络流动，每张卡只处理流经它的那一小块数据，算完就传给下一张卡，绝不囤积数据。

> [图片 1：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

这个过程会进行轮 (Steps)， 对应环上的 P 个节点。 1 ． 2 ． 3 ． lnner Loop （计算）： GPU 使用本地的 Q 和当前持有的 Key/Value 块 （记为 Kcurr， Vcurr) 计算注意力分数和局部输出。 · 计算使用 FlashAttention 的逻辑 (Blockwise)， 维护局部的 Softmax 归一化因子 (Log- Sum-Exp 统计量）。 Communication （通信 - 关键步骤）： · 在计算的同时， GPU 将当前的 K 发送给下一张卡 (GPU 犭 + 1）。 CtLT•r， CtLrr · 同时， GPU 从上一张卡 (GPUi— 1） 接收新的 Key/Va | u e 块。 OverIap （计算与通信重叠）： 这是 Ring Attention 的精髓。 由于矩阵乘法 （计算） 非常耗时， 而传输 K, V 块 （通信） 相对较快。 系统设计使得在计算第块数据的同时， 网络已经在传输第 + 1 块数据。 如果计算时间 > 传输时间， 通信延迟就被完全隐藏了 (Zero-overhead communication)。

其中计算依赖于Softmax的分块计算性质：

> [图片 2：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

Ring Attention 之所以能成立， 依赖于 Softmax 的分块计算性质 （也是 FlashAttention 的基础）。 标准的 Softmax 需要全局的分母： Softmax(c)t 但是， 我们可以把分母拆开累加。 假设序列被切分成两块和 B： 1 ． 先算块的局部最大值 mA 和局部指数和 = j 4 2 ． 再算块 B 的局部最大值 rnB 和局部指数和 IB = 一 7 召 jeB 3 ． 最后可以通过简单的数学变换将两者合并， 更新全局的 m 和 lglobal， 而不需要同时持有块和块 B 的原始数据。 Ring Attention 就是利用这个原理， 每次只在显存里放一个 Block 的 K, V, 算完更新统计量， 然后扔掉 （或传走） 该 Block0

计算的具体步骤如下：

> [图片 3：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

Q， 人 《 计算分数： S = 局部统计量 · 计算当前块的行最大值 m = max(S) 计算当前块的指数和 lblock = E exp （S 一 mblock 计算非归一化的注意力输出 Oblock = exp （S 一 mblock)Vcurr

> [图片 4：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

Q， 人 《 计算分数： S = 局部统计量 · 计算当前块的行最大值 m = max(S) 计算当前块的指数和 lblock = E exp （S 一 mblock 计算非归一化的注意力输出 Oblock = exp （S 一 mblock)Vcurr

第i张显卡的输出O是一个和Q_i行数相同的矩阵（如果每张显卡只存一个Query向量，则O就是一个向量），是对于每个Query，根据目前已经经过该显卡的K、V得到的对V进行注意力加权的结果。每一轮计算后，更新统计量和输出O：

> [图片 5：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

更新全局最大值 · 更新归一化因子： prev—mnew netV max 7 mblock prev， —mnew mblock 更新输出 0 （核心重缩放公式）： pret， block · lblock —mnew block

## 参考文献与引用线索

> 本节由脚本自动检索正文中的引用线索，可能不完整；未能确定来源的位置会在下方标为待补引用。

### 待补引用或版权检查

- [待补引用] 本文含 Word 内嵌图片；开源版未上传图片。若图片来自教材、论文或技术报告，建议人工确认授权、补充来源或重画。
- [待补引用] 未自动检索到明确参考文献线索，建议人工补充可追溯来源。
