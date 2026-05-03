---
title: "20.1 Multi-token Prediction"
source_docx: "第3部分 大语言模型/20.大模型的架构和训练方法优化/20.1 Multi-token Prediction.docx"
status: "auto-converted"
ocr: "auto-generated, needs human review"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 20.1 Multi-token Prediction

> 本文由本地 Word 原稿自动转换而来。图片中的文字由 OCR 自动识别，可能存在识别错误，欢迎提交 Issue 修正。

## 一、Multi-Token Prediction（MTP）的提出背景

由于LLM的推理是自回归的，每次生成一个token，都要在显存中加载一轮完整的模型参数，导致显存带宽成为推理速度的显著瓶颈。同时，在利用Next token prediction进行训练的过程中，模型只能学习基于当前预测下一个token的质量，而无法拥有更远的“视野”。Multi-Token Prediction（MTP）通过在训练和推理中每一次并行生成多个token，来解决这些问题。

## 二、Blockwise Parallel Decoding

这是Google于2018年提出的范式，也是MTP的初始形态。

1.模型架构

> [图片 1：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

Head 1 (original model) vocab—predict—head h x vocab—size Head2 (auxiliary models 2） vocab—predict—head h × vocab—size Headk (auxiliary models k) vocab—predict—head h x vocab—size 参数共享残差连接 FFN FFN logit 4h × h Last Hidden (h) FFN FFN headfi 入 4h x h h x 4h FFN FFN 4h × h h x 4h 每个 Head 爹数不同 Transformer Layers 主干网络 embedding

如图，主干网络是训练好的多层decode-only的Transformer网络，经过多层前向计算后，最终隐藏层输出h维度（＝embedding维度）的logit。上面接了多个输出Head，每个Head负责预估一个token。每个Head有三层：首先是一个共享的FFN层，将logit做宽映射（h维->4h维）；然后再过一个非共享的FFN层，将logit维度还原（4h维->h维），经残差连接，得到h维embedding向量。最后，再将结果送入到词表投影层得到每个词的概率分布。

2.生成范式

> [图片 2：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

Predict Verify I I saw saw a a a a dog dog dog dog ride ride ride nde ln ln ln ln the the the bus car executed in parallel

MTP的生成过程是一个“Predict-Verify”的循环过程。先一次性预测K个token，然后利用Transformer的并行性，如图通过掩码的方式实现并行验证。如果预测全部正确，相当于用2次推理的时间实现了K次推理。

进一步地，重叠第n步的verify阶段和第n+1步的predict阶段，能进一步提高推理性能。如图，先预测出3个token，然后在验证阶段，仍然每次都预测3个token。如由于第2个正确，可以以其为条件生成第3个“car”、第4个“this”和第5个“week”，由于最初预测的第3个对不上，故最初的预测只能留下“in”和“the”，然后这里生成的第3~5个就可以直接作为新的一轮预测，后续再验证此时生成的第4和第5个，以此类推，不需要重新预测3个token。

> [图片 3：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

Predict ve rify （+ next Predict) Predict I I I I I S aw saw S aw S aw saw a a a a a dog dog dog dog dog n de nde 巧 de ride de m m ln ln ln the the the the the bus car C ar bus car last this last thi S exec uted week in p a11e1 week When week

本质上，这就是利用多token生成的并行性，把后续根据“in”“the”这两个正确的token进行新一轮预测的步骤并入之前的验证步骤。

## 三、Meta's MTP

如图，Meta让每个头不仅仅是FFN层，还有Transformer层，从而可以处理更复杂的序列上下文关系。

> [图片 4：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

0 CrossEntropyLoss 参致共享 Head1 vocab-predict—head FFN 2 MHA Transformer logit (original model) h × vocab_size h x h Last Hldden (h) Head2 (auxiliary models 2） vocab-predlct—head h × vocab—size FFN · 2 h × h M HA Transformer head$ü 入 Headk (auxiliary models k) vocab—predict—head h x vocab—stze FFN ． 2 h x h MHA Transfomer Transfomer Layers 主干网络 embedding

## 四、DeepSeek MTP

> [图片 5：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

> [图片 5 OCR 未识别出有效文字：OCR did not return a result.]

> [图片 6：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

一条样本序列长度为 T = 10， 默认增加 s 位以 input Main Model 样本枸建 label input MTP Module 1 样本枸建 input MTP Module k 样本枸建 label 0 tl tl tl tl 1 惚 t2 t2 惚相隔 k 步 2 t3 t3 13 t3 3 14 t4 4 t5 t5 t5 1 + k 十 1 5 t6 t6 6 t7 7 t8 t8 t8 t8 t8 8 t9 t9 t9 t9 t9 9 tl 0 tl 0 tl 0 t10 eos eos eos eos

训练阶段：Main Model：由t1生成t2；由t1，t2生成t3；……；由t1~t10生成eos token；计算平均Cross-Entropy Loss。MTP Model 1：MTP Module：由t1，t2生成t3；……；由t1~t9生成eos token；计算平均Cross-Entropy Loss。后续的MTP Module以此类推。

> [图片 7：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

训练 CrossEntropyLoss 0 数共享 Head1 vocab-predict—head logit (main model) h × vocab_stze Last Hidden 间 Head2 (MTP ModuIe 1） vocab—predict—head h x vocab—size Transformer h × h Linear Projection 2h × h RMSNorm RMSNorm embedding Headk (MTP Module k) vocab—predict—head h x vocab—size Transformer Layers 主干网络 embedding Transformer h x h Linear Projection 2h × h R MSNorm RMSNorm embedding 参数共！ Teacher forcing 式训伍褸萝 ou 斷， 刀匝孬 “ 阼力的入） ·

从模型架构上看，DeepSeek在Meta工作的基础上，还在MTP的Transformer层前加上了额外的输入。这个额外输入在训练时是ground truth的t2和t3，以防止细微误差导致“跑偏”；在推理时则是模型自身预测的t2和t3（虽然运用了上一次预测，但这里并非退化为Next token prediction，因为自身预测的t2和t3只经过轻量级MTP模块，不经过全模型，故仍为MTP）。MTP头的损失：

> [图片 8：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

= CrossEntropy(P2+k•T+1' t2+k:T+1) MTP T+I 〗 log pk 圄， 忙 2 + k

DeepSeek原论文中插图如下：

> [图片 9：原 Word 此处有图片；为避免版权风险，开源版暂不上传图片。]

**图片文字 OCR（自动识别，待校对；数学公式必须人工核对）：**

> [公式校对警告] 这段 OCR 文本疑似包含数学公式、上下标、希腊字母或高密度符号。请不要直接把自动识别结果视为可靠公式。

Target t2 ts t3 Cross-Entropy LOSS MTP Module 1 “ t2 丆 nPre “ 丿 Output Head Transformer Block Linear Projection concatenation RMSNorm RMSNorm Embedding Layer 坛 t7 Cross-Entropy LOSS Main Model Out ut Head Transformer B10 × L Em bedding Layer lnput Tokens Cross-Entropy LOSS MTP Module 2 O utput Head Transformer Block Linear Projection conca 冂 0 0 RMSNorm RMSNorm Embedding Layer MTP

## 参考文献与引用线索

> 本节由脚本自动检索正文中的引用线索，可能不完整；未能确定来源的位置会在下方标为待补引用。

### 自动检索到的引用线索

- DeepSeek原论文中插图如下：

### 待补引用或版权检查

- [待补引用] 本文含 Word 内嵌图片；开源版未上传图片。若图片来自教材、论文或技术报告，建议人工确认授权、补充来源或重画。
