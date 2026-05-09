---
title: "34.1 自回归、扩散的结合与时空Transformer"
source_docx: "第5部分 世界模型、多模态生成与具身智能/34.多模态生成与生成式世界模型/34.1 自回归、扩散的结合与时空Transformer.docx"
status: "auto-converted"
ocr: "image placeholders rebuilt as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 34.1 自回归、扩散的结合与时空Transformer


## 一、自回归生成的世界模型

### （一）自回归生成范式：以Genie 1为例

下面以Genie 1为例，介绍自回归生成的经典范式，及其核心架构（如时空Transformer等）。

训练阶段（Training Phase）：

1. 首先，单独训练视频分词器（Video Tokenizer），实现对原始视频流的压缩。

2. 然后，联合训练潜在动作模型（LAM）和动态模型（Dynamics Model）。LAM 直接从像素视频中提取潜在动作，动态模型则利用视频 Token 进行预测。

推理/交互阶段（Inference/Play Phase）：

1. 用户提供一张初始提示图像 $x_1$ 作为首帧。

2. 分词器将该图像转换为初始 Token $z_1$。

3. 用户输入一个离散动作指令 $a_1\in[0,|\mathcal{A}|)$。

4. 系统在 VQ Codebook 中检索动作 $a_1$ 对应的嵌入表示 $\bar{a}_1$。

5. 动态模型结合 $z_1$ 和 $\bar{a}_1$ 预测下一帧的 Token $z_2$。

6. 分词器的解码器将 $z_2$ 还原为像素图像展现给用户，并不断循环以生成连续的可控视频。

其中，若用户还输入了其他额外指令或存在其他上下文 $c_1$，则将其经过投影后，与 $z_1$、$a_1$ 拼接进行自注意力，或与 $z_1$、$a_1$ 之间进行交叉注意力。

模型架构：

视频分词器（Video Tokenizer）：用于将高维视频帧序列降维压缩为离散的 Token 序列。给定视频帧 $x_{\le t}$，分词器将其编码为离散表示 $z_{\le t}$。由于编码器和解码器都使用 ST-Transformer 结构，每一个离散编码都融合了过去所有帧的时序动态信息。

潜在动作模型（Latent Action Model, LAM）：该组件是实现无动作标签训练的关键。编码器接收历史帧 $x_{\le t}$ 和下一帧 $x_{t+1}$，推断它们之间的连续动作嵌入 $\tilde{a}_t$，随后通过 VQ Codebook 将其量化为离散动作 $a_t$。在 Genie 1 中，动作字典大小限制为 $|\mathcal{A}|=8$，以确保人类可玩性和控制力。最后，解码器仅根据历史帧序列和潜在动作来重建并预测下一帧 $\hat{x}_{t+1}$。这种预测瓶颈迫使模型提取能够引起帧间实质性变化的核心动作信息。

动态模型（Dynamics Model）：这是一个基于 MaskGIT 的自回归解码器模型。在时间步 $t$，它接收过去的视频 Token $z_{\le t-1}$ 以及对应的潜在动作嵌入 $\bar{a}_{\le t-1}$，并预测下一帧的 Token。模型通过预测 Token 与真实 Token 之间的交叉熵损失进行训练。训练时采用伯努利分布对输入 Token 随机掩码，掩码率在 0.5 到 1 之间均匀采样。
关于视频分词器生成的表示：在自回归模型中，一般用离散token ID构成的序列（不同于扩散模型每一个token为高维向量）。

动态模型中过去的视频和动作嵌入如何融合：

视频帧经过分词器（Video Tokenizer）后，会去查分词器的 VQ 码本。论文中提到分词器的码本嵌入维度（latent dim）为 32。动作指令（离散索引）会去查 LAM 的 VQ 码本，其码本嵌入维度同样是 32。
可以看出，Genie实现互动性是一大进步，但动作仍然局限在少量的可选范围内，远未达到开放真实世界的水平。

动态模型本身有一个很大的隐藏层维度。例如在 0.1B 的 Genie 模型中，$d_{\mathrm{model}}$ 达到 5120。为了进入 Transformer，视频 Token 嵌入和动作嵌入都会分别通过线性层投影，从 32 维投影到 5120 维。

很多早期世界模型会将动作拼接到视频特征后面（Concatenation），这会导致序列长度增加或特征维度翻倍。Genie 的作者发现，将动作作为 **加性嵌入（Additive Embeddings）** 直接与对应的视频 Token 相加，既能保持维度不变，又能显著提升视频生成的可控性：

$$
E_{\mathrm{input}}=E_{\mathrm{video}}+E_{\mathrm{action}}+E_{\mathrm{position}}
$$

这里还会加上标准的位置编码 $E_{\mathrm{position}}$。

### （二）Multi-token Prediction

基于自回归生成的世界模型存在着生成速度慢的问题，但在具身智能等领域，这种延迟会对机器人的灵活性造成较大的影响。为了提高推理速度，可以采用Multi-token prediction的方法，一次生成多个token后并行验证。

## 二、自回归与扩散的结合范式

### （一）块内扩散、块间自回归

这是自回归与扩散结合最常见的范式，即用同一个网络，每次用扩散方法生成一个块，每次扩散生成的块会被存在KV Cache中，块间用自回归方式。

### （二）自回归预测与扩散生成的解耦

Nvidia在Cosmos中，提出在自回归预测的基础上，运用扩散提高生成质量的方法。和上面不同的地方在于，这里自回归和扩散在两个网络中进行，用自回归网络预测出的块再给扩散网络进行视觉生成，自回归网络每预测一个块时，看到的上下文（即KV Cache中的递归对象）是自回归网络自身而非扩散网络生成的块。

自回归模型使用离散分词器，能够以高压缩率把海量视频信息压缩为少数整数，但这种激进压缩会导致解码出的视频出现模糊或伪影。相比之下，扩散模型使用的连续分词器能保留更多细节。为了弥补这一缺陷，Cosmos 微调了一个 7B 的文本到视频扩散模型（Cosmos-Predict1-7B-Text2Video），将其改造为一个强大的“解码器”。

训练阶段：使用离散分词器的自回归模型输出一个模糊的目标图像，并将其长宽放大2倍以使其像素数与最终清晰的目标图像一致。我们用一个扩散模型，根据扩散模型的训练原理，我们需要给清晰的目标图像加噪作为训练的初始状态，以模糊的目标图像作为上下文，输出去噪后清晰的目标图像，根据与实际的差距计算梯度训练。

推断工作流：

1. 自回归模型生成高度压缩的“离散 Token 视频”。

2. 将其转化为条件输入，通过微调好的扩散去噪网络（Diffusion Denoiser）上采样并预测出更清晰的“连续 Token 视频”。

3. 最后，将这些连续 Tokens 送入 Cosmos 连续分词器的解码器，还原出更清晰的 RGB 物理世界视频。
### （三）扩散模型的级联

可以用一个块内扩散、块间自回归的模型生成低分辨率视频，然后一方面上采样并给扩散模型生成可用于解码的高分辨率视频，一方面将低分辨率视频存入KV Cache同步进行下一步生成，可能可减少延迟。

## 三、时空注意力

时空 Transformer（ST-Transformer）：传统视频 Transformer 的计算复杂度呈二次方增长，而 ST-Transformer 将空间注意力和时间注意力解耦。在每个 ST 块中，空间注意力仅在单一时间步内的 $1\times H\times W$ 个 Token 上计算，时间注意力则在跨越 $T$ 个时间步的 $T\times 1\times 1$ 个相同空间位置 Token 上计算。这使得核心计算成本随帧数更接近线性增长，而不是直接对全体 $T\times H\times W$ 个 Token 做全局注意力。
注意这里的H*W不是一帧的像素个数（甚至可能不是token个数），而是一帧的分块个数。每个分块的向量维度会经过输入投影层压缩。时空Transformer具体架构如下：

第一步：空间特征交互。输入的 Token 首先进入空间注意力层（Spatial Attention），只与同一帧内的其他 Token 交换信息。

第二步：时间特征交互。经过空间层更新后的 Token 作为输入进入时间注意力层（Temporal Attention），与跨越不同帧但处于同一空间位置的 Token 交换信息。
### （一）空间注意力

对于任意给定的第 $t$ 帧，其对应的 Token 矩阵为 $Z_t\in\mathbb{R}^{S\times D}$。在这个切片上，计算标准自注意力：

$$
Q_t^S=Z_tW_Q^S,\quad K_t^S=Z_tW_K^S,\quad V_t^S=Z_tW_V^S
$$

$$
\mathrm{SpatialAttention}(Z_t)=\mathrm{Softmax}\left(\frac{Q_t^S(K_t^S)^\top}{\sqrt{d_k}}\right)V_t^S
$$

这里，$W_Q^S,W_K^S,W_V^S\in\mathbb{R}^{D\times d_k}$ 是空间注意力层独有的可学习权重矩阵。在这个过程中，每个 Token 只与同一时间步（同一帧）内的 $1\times H\times W$ 个 Token 计算注意力分布。

复杂度分析：对于单帧，复杂度是 $O(S^2)$。因为有 $T$ 帧并行计算，总计算复杂度为 $T\times O(S^2)=O(TS^2)$。此时，占据计算资源主导地位的空间注意力层，其计算复杂度已经降为随帧数 $T$ 呈线性增长。
对于存在视觉、文本两类token混合的情况，以Seedance为例：

双流架构（MMDiT Design）：借鉴 Stable Diffusion 3 的设计，空间层采用多模态自注意力（Multi-Modality Self-Attention）整合视觉和文本 Token。

模态隔离的权重参数：由于视觉 Token（来自 VAE）和文本 Token（来自经过微调的 LLM）在特征分布和语义空间上存在很大差异，如果在同一空间内强制共享权重，容易导致优化冲突。因此，Seedance 在空间层为两种模态保留两套完全独立的网络权重，包括自适应层归一化（Adaptive Layer Norm, AdaLN）、$Q,K,V$ 的投影矩阵（Projection）以及多层感知机（MLP）。
### （二）时间注意力

时间注意力仅对视觉token计算，文本token不参与。在每一帧内部执行窗口划分，每个token只与过去时间段内位于同一个窗口的token交互。（下面为了简化，默认一个窗口仅1个token）

对于空间上的任意固定位置 $s$，其在时间轴上的 Token 矩阵为 $Z_s\in\mathbb{R}^{T\times D}$。在此切片上计算时间维度的自注意力：

$$
Q_s^T=Z_sW_Q^T,\quad K_s^T=Z_sW_K^T,\quad V_s^T=Z_sW_V^T
$$

其中自回归生成时：

由于视频具有时间先后顺序，特别是在做自回归预测时，时间层必须引入因果掩码（Causal Mask）$M$：

$$
\mathrm{TemporalAttention}(Z_s)=\mathrm{Softmax}\left(\frac{Q_s^T(K_s^T)^\top}{\sqrt{d_k}}+M\right)V_s^T
$$

矩阵 $M$ 是一个上三角矩阵，对角线以上的元素为 $-\infty$，对角线及以下的元素为 0。这保证了在计算第 $t$ 帧的特征时，模型只能“看”到自身和过去的帧，不能“偷看”未来帧。在这个过程中，每个 Token 只与跨越 $T$ 个时间步的同一空间位置的 $T\times 1\times 1$ 个 Token 计算注意力。
对于“块内扩散，块间自回归”的范式，根据块级掩码分析即可。

复杂度分析：单位置的时间注意力复杂度为 $O(T^2)$。由于有 $S$ 个空间位置，总复杂度为 $S\times O(T^2)=O(ST^2)$。

## 参考文献

- Kondratyuk, D., Yu, L., Gu, X., et al. (2023). [VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://arxiv.org/abs/2312.14125). arXiv:2312.14125.
- Villegas, R., Yang, J., Hong, S., et al. (2022). [Phenaki: Variable Length Video Generation from Open Domain Textual Description](https://arxiv.org/abs/2210.02399). arXiv:2210.02399.
- Peebles, W., & Xie, S. (2023). [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). ICCV.
- Bruce, J., Dennis, M., Edwards, A., et al. (2024). [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). ICML.
