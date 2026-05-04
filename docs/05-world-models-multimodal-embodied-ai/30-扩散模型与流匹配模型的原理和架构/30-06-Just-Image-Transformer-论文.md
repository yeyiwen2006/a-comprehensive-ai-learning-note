---
title: "30.6 Just Image Transformer（论文）"
source_docx: "第5部分 世界模型、多模态生成与具身智能/30.扩散模型与流匹配模型的原理和架构/30.6 Just Image Transformer（论文）.docx"
status: "auto-converted"
ocr: "disabled; image content awaits manual reconstruction"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 30.6 Just Image Transformer（论文）


> 本文是论文阅读笔记，内容代表对应论文方法或作者理解，不应直接视为领域共识或工程最佳实践。

## 一、核心思想

当今主流的扩散模型（如 DDPM）和流匹配模型（Flow Matching）通常训练神经网络去预测噪声 $\epsilon$ 或速度 $v$。JiT 指出，预测干净数据（x-prediction）与预测含噪量（$\epsilon$ 或 v-prediction）在数学本质上是完全不同的。

这一结论建立在**流形假设（Manifold Assumption）**之上：

- **干净图像 $x$**：高维像素空间 $\mathbb{R}^D$ 中的自然图像并非均匀分布，而是高度集中在一个极低维的流形 $\mathcal{M}$ 上（$d\ll D$）。
- **噪声 $\epsilon$**：高斯噪声是各向同性的，它游离于流形之外（off-manifold），散布在整个高维空间 $\mathbb{R}^D$ 中。
- **速度 $v$**：在流匹配中，$v=\epsilon-x$，同样是高维空间中游离于流形之外的量。

当网络被要求预测 $\epsilon$ 或 $v$ 时，它被迫去拟合一个覆盖整个高维空间的无结构映射。而当网络被要求直接预测 $x$ 时，其预测目标被严格限制在低维的自然图像流形 $\mathcal{M}$ 上。这种目标的降维极大地降低了神经网络的拟合难度，使得模型能够在不依赖 Latent 空间降维的情况下，直接在高分辨率像素空间（如 $512\times512$）中高效运作。
## 二、模型架构

JiT 的架构可以说是“除了标准的图像 Transformer，什么都没有”。

1. **无 Tokenizer 与无 Latent 空间**：完全抛弃了主流架构中使用的 VAE 预训练模型，直接对原始像素（Raw Pixels）进行操作。
2. **大图像块（Large Patch Size）**：为了处理高分辨率图像带来的序列长度爆炸，JiT 采用了非常大的 Patch Size（如 $16\times16$ 甚至 $32\times32$ 和 $64\times64$）。
3. **信息瓶颈（Bottleneck Design）**：实验发现，在 x-prediction 模式下，大幅压缩 Transformer 的线性嵌入维度（Embedding Dimension）不仅不会导致模型崩溃，反而能保持鲁棒性。这从侧面印证了流形假设：因为目标 $x$ 本质是低维的，所以低容量（“under-capacity”）的网络瓶颈足以捕获其特征；而预测高维噪声 $\epsilon$ 时，缩小网络容量则会导致灾难性失效。
4. **无预训练与无额外损失**：不需要任何形式的预训练，也不依赖感知损失（Perceptual Loss）或对抗损失（Adversarial Loss）。
唯一新增的模块是在Transformer后加的一个全连接层，把生成的低维嵌入向量映射为高维的图像。这样做有效的原因在于，真实世界的高维图像斑块并不是随机杂乱的，它们具有极强的空间连贯性和规律，因此它们被紧紧约束在一个极低维的“流形”上。Transformer内部的低维隐变量已经足以捕捉这些低维的结构信息，最后的那个全连接层，本质上只是把这个低维结构“旋转”并“映射”回高维空间中对应的流形位置而已。

时间步和Diffusion Transformer一样，通过自适应层归一化的方式注入。

1. **时间步嵌入（Time Embedding）**：

   首先，标量时间步 $t$ 会通过正弦位置编码（Sinusoidal Positional Encoding）映射为高维向量，然后再通过一个多层感知机（MLP）提取特征，生成全局的时间特征向量 $E_t$。

2. **生成调制参数（Modulation Parameters）**：

   每个 Transformer Block 外部设有一个简单的线性回归层（Linear Layer）。该层接收 $E_t$ 作为输入，并直接回归出该 Block 所需的几组调制参数（通常是缩放因子 $\gamma$、平移因子 $\beta$ 以及用于残差连接的门控因子 $\alpha$）。

3. **自适应调制（Adaptive Modulation）**：

   在输入的视觉 Token 序列进入多头自注意力层（MSA）或前馈神经网络（FFN）之前，先对 Token 进行标准的 Layer Normalization。随后，使用刚才回归出的 $\gamma$ 和 $\beta$ 对归一化后的特征进行逐元素的仿射变换：

$$
\mathrm{AdaLN}(x,t)=\gamma(t)\cdot\mathrm{LayerNorm}(x)+\beta(t)
$$

在这里，归一化的尺度和偏移完全由当前的时间步 $t$ 动态决定。
> [图片内容待重建：img-a5089c6b0ba4-0004] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
## 三、既然如此，为什么业界仍然使用原始扩散模型？

1.潜在空间已经避免了维度灾难

> [图片内容待重建：img-a5089c6b0ba4-0005] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
> [图片内容待重建：img-a5089c6b0ba4-0006] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
2.动作空间的多样性

> [图片内容待重建：img-a5089c6b0ba4-0007] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
> [图片内容待重建：img-a5089c6b0ba4-0008] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
注意：这里的p(x)不是训练数据的联合概率密度，而是对于新数据本身的概率分布。

3.高噪声条件下的稳定性

> [图片内容待重建：img-a5089c6b0ba4-0009] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。
> [图片内容待重建：img-a5089c6b0ba4-0010] 原 Word 此处有图片。为避免版权风险，开源版暂不上传图片；自动 OCR 已弃用，后续将依据原稿人工重建为 Markdown/LaTeX。

## 参考文献

暂无已核验参考文献。
