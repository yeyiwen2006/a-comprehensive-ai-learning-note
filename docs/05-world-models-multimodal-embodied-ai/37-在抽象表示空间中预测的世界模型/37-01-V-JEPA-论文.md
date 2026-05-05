---
title: "37.1 V-JEPA（论文）"
source_docx: "第5部分 世界模型、多模态生成与具身智能/37.在抽象表示空间中预测的世界模型/37.1 V-JEPA（论文）.docx"
status: "auto-converted"
ocr: "disabled; image content awaits manual reconstruction"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 37.1 V-JEPA（论文）


> 本文是论文阅读笔记，内容代表对应论文方法或作者理解，不应直接视为领域共识或工程最佳实践。

## 一、核心思想

在 V-JEPA 之前，主流的视频无监督学习通常依赖于像素级重建（Pixel Reconstruction），例如掩码自编码器 VideoMAE。然而，在像素空间进行预测存在明显劣势：模型必须分配大量计算能力和模型容量来捕捉视觉输入中那些低层次、甚至是无关紧要的细节。

V-JEPA 的核心思想是特征空间预测（Feature Prediction）：

- 该模型完全摒弃了像素重建，不再预测原始像素，而是预测被遮挡区域在语义特征空间中的表示。
- 通过在隐性表示空间（Latent Space）中进行预测，模型能够灵活地消除视频中不可预测或与语义无关的像素级细节，从而学习到高度通用的视觉特征。

其他基于隐变量的方法更像Decoder，按时序生成下一时刻的隐变量；V-JEPA更像Encoder，在时空上进行掩码预测，且没有重建损失（即由隐变量生成画面或间接参与生成画面）。

## 二、核心架构与损失函数

V-JEPA 的核心架构由三个部分组成：编码器 $E_\theta(\cdot)$、预测器 $P_\theta(\cdot)$ 以及目标编码器 $E_{\theta'}(\cdot)$。目标是让从视频某一部分 $x$ 计算出的表示，能够预测视频另一部分 $y$ 的表示。

为了防止模型输出常数向量从而陷入“表示坍塌”（Representation Collapse），V-JEPA 借鉴了非对比学习策略，采用了带有停止梯度（Stop-gradient）的指数移动平均（EMA）架构。

模型的优化目标是最小化预测特征与目标特征之间的 $L_1$ 距离：

$$
\min
\left\|
P_\theta(E_\theta(x),\Delta_y)
-
sg(E_{\theta'}(y))
\right\|_1
$$

其中，$sg(\cdot)$ 表示停止梯度操作，$E_{\theta'}(\cdot)$ 是 $E_\theta(\cdot)$ 权重的指数移动平均网络，$\Delta_y$ 是提供给预测器的位置条件变量。

解释：

1.用L1范数是因为受极端值影响小，实验发现更稳定。其仅在0处不可导（类似ReLU）。

2.由于没有重建损失，为了防止“模式崩塌”，y的Encoder更新方法类似于强化学习的“目标网络”，即从x的Encoder网络软更新：

EMA 更新（y-encoder）：y-encoder 的权重 $\theta'_t$ 不使用优化器更新，而是使用指数移动平均公式缓慢更新：

$$
\theta'_t=\tau\theta'_{t-1}+(1-\tau)\theta_t
$$

其中 $\tau$ 是动量参数，在 V-JEPA 中从 $0.998$ 逐渐增加到 $1.0$。也就是说，y-encoder 始终是 x-encoder 历史状态的一个平滑、滞后的影子。

为什么这样能防止模式崩塌：

假设我们固定住编码器，只训练预测器 $P_\theta$。模型的损失函数是 $L_1$ 范数（绝对误差）。在统计学中，使得绝对误差期望最小的预测值，正是目标分布的中位数。

当预测器已经训练得非常完美（即 $P=P^*$）时，我们将它代回损失函数，这时候整个损失函数就变成了计算目标 $Y$ 的条件中位绝对偏差（Median Absolute Deviation，简称 MAD）：

$$
MAD(Y\mid E_\theta(x))
=
\mathbb{E}
\left[
\left|
Y-\mathrm{median}(Y\mid E_\theta(x))
\right|
\right]
$$

此时，编码器 $E_\theta$ 的优化方向（梯度）就是去最小化这个 MAD。

直观解释：

- 如果 $E_\theta(x)$ 抽取了“什么都没学到、输出常数”，那么它就无法提供关于目标 $Y$ 的任何线索。此时 $Y$ 在给定 $E_\theta(x)$ 条件下的分布会非常分散，其 MAD 值会非常大。
- 为了最小化 MAD，编码器 $E_\theta(x)$ 必须从输入视频中抽取有效信息，使得它能够高度确定地推出 $Y$ 的状态。当 $E_\theta(x)$ 包含的信息越丰富，目标 $Y$ 的不确定性就越小，分布越集中，MAD 也越小。
- 虽然最小化 MAD 能促使编码器学习，但如果目标 $Y$（由 y-encoder 生成）和预测器更新得一样快，它们可能会“合谋”一起变成常数来让 MAD 变为 $0$。
- 这就是 EMA 的原因：y-encoder 的权重不是通过梯度下降更新的，而是 x-encoder 权重的历史平均版本。这使得 y-encoder 的变化非常缓慢。

工作流：

1. 预测器学得很快，迅速逼近最优解 $P^*$。
2. 因为目标 $Y$ 变化很慢，它不会配合 x-encoder 一起坍塌。
3. 迫于快速进化的预测器和稳定缓慢的目标，x-encoder 只能老老实实地去学习如何提取丰富的信息来降低 MAD，从而从根本上切断了坍塌的可能。

这意味着通过软更新的方式把在线网络的参数复制给目标网络，可以让目标网络的变化总是落后于E_θ(x)，且目标网络只能被动随着在线网络变化而变化，就像在线网络在追一个“移动的靶子”，避免了双方共同奔向常数的捷径。

## 三、训练方法

### （一）掩码token预测

- 模型首先从视频中抽取一个包含 $16$ 帧的短视频片段（时间步幅为 $4$），并将其空间分辨率调整为 $224\times224$，张量形状为 $16\times224\times224\times3$。
- 使用 3D 卷积层（过滤器大小 $2\times16\times16$）对视频进行切块，将其展平为一维的词元（Token）序列，序列长度为 $1568$。
- 将绝对 3D 正弦-余弦位置嵌入（Position Embeddings）添加到该特征图中。
- 采用多块采样策略：提取短程掩码（覆盖 $15\%$ 面积的 $8$ 个随机块）和长程掩码（覆盖 $70\%$ 面积的 $2$ 个随机块）。平均掩码比例高达约 $90\%$。

> [图片内容待重建：img-dee43f1446a8-0011] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
> [图片内容待重建：img-dee43f1446a8-0012] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
为什么不严格按时序因果：

> [图片内容待重建：img-dee43f1446a8-0013] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
### （二）非掩码token的损失函数

在V-JEPA 2.1中，为了更充分地利用数据以及让模型更好地理解连续性，对非掩码token也引入了损失，要求预测这些未被遮挡的特征：

> [图片内容待重建：img-dee43f1446a8-0014] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
> [图片内容待重建：img-dee43f1446a8-0015] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
这样设计的好处：

> [图片内容待重建：img-dee43f1446a8-0016] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
### （三）深度自监督

> [图片内容待重建：img-dee43f1446a8-0017] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
> [图片内容待重建：img-dee43f1446a8-0018] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
> [图片内容待重建：img-dee43f1446a8-0019] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
## 四、与LLM的结合

预训练完成后，我们得到了一个不懂人类语言，但对物理世界、物体外观和运动规律有着极深理解的模型。在V-JEPA 2和LLM之间加入一个投影器（Projector，通常是多层感知机MLP），把V-JEPA 2提取出的纯视觉特征（视觉Token），“翻译”并映射到大语言模型能够理解的输入空间（文本嵌入空间）中。然后用大规模的“图像/视频-文本”问答对齐数据来训练这个整体系统，就可以让其逐步学会了如何理解由V-JEPA 2传递过来的视觉信息，并能通过LLM的处理用自然语言回答关于视频内容的问题，实现初步的空间推理。

## 参考文献

- Bardes, A., Garrido, Q., Ponce, J., et al. (2024). [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471). arXiv:2404.08471.
- Assran, M., Duval, Q., Misra, I., et al. (2023). [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243). CVPR.
