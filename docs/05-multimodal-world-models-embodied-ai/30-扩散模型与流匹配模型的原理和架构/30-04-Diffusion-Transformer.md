---
title: "30.4 Diffusion Transformer"
source_docx: "第5部分 多模态生成、世界模型与具身智能/30.扩散模型与流匹配模型的原理和架构/30.4 Diffusion Transformer.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 30.4 Diffusion Transformer


## 一、核心架构与工作流

Diffusion Transformer将Transformer架构引入扩散模型，实现了与大语言模型架构的统一，也提高了可扩展性，使得生成质量可随着模型参数的扩大而提升。

1. **潜在空间分块（Latent Patchification）**

DiT 并不直接操作像素，而是操作 VAE（变分自编码器）编码后的潜在特征。

给定一张输入图像 $x\in\mathbb{R}^{H\times W\times 3}$，首先通过预训练的 VAE 编码器将其映射到潜在空间，得到潜在表示 $z\in\mathbb{R}^{h\times w\times c}$。

接着，DiT 将 $z$ 切分为大小为 $p\times p$ 的空间块（Patches），将这些块展平并经过线性投影，转换为序列长度为 $T=(h/p)\times(w/p)$、维度为 $d$ 的 Token 序列。最后，加上标准的可学习位置编码（Positional Embeddings）。

2. **条件注入机制：adaLN-Zero Block**

标准的扩散模型需要在每个去噪步骤中注入当前的时间步 $t$ 和条件类别 $c$。U-Net 通常通过交叉注意力（Cross-Attention）或特征拼接来实现，而 DiT 采用了自适应层归一化（Adaptive Layer Normalization, adaLN）。

这是 DiT 架构中最关键的设计。对于传入 DiT Block 的隐藏层状态 $h$，adaLN 的操作如下：

$$
\mathrm{adaLN}(h,t,c)=\gamma_{(t,c)}\mathrm{LayerNorm}(h)+\beta_{(t,c)}
$$

其中，缩放参数 $\gamma_{(t,c)}$ 和平移参数 $\beta_{(t,c)}$ 是通过对时间步嵌入和类别嵌入的加和应用一个多层感知机（MLP）回归直接生成的，而不是作为可学习参数固定在网络中。

此外，DiT 还引入了**门控机制（Dimension Scaling）**，用于残差连接之前。为了极大地稳定训练，作者提出了 **adaLN-Zero** 初始化策略：即强制 MLP 在初始化时输出的 $\gamma_{(t,c)}$、$\beta_{(t,c)}$ 以及残差门控参数 $\alpha_{(t,c)}$ 全部为零。

这意味着在训练初始阶段，**每一个 DiT Block 都是一个严格的恒等映射（Identity Function）**：

$$
h_{l+1}=h_l+0=h_l
$$

这保证了网络在初始状态下不会因为深层 Transformer 的高方差而崩溃。

3. **Unpatchification（去块化与预测）**

在经过 $N$ 层 DiT Block 后，Token 序列被送入最后一个标准的全连接层，将维度扩张回 $p\times p\times 2c$（乘以 2 是因为模型需要同时预测噪声和方差）。随后，将一维序列重新排列（Reshape）回 $h\times w\times 2c$ 的空间网格结构，输出给后续的损失函数计算或采样。
我们发现adaLN只能注入定长向量 $c$，如 `[CLS]` token。但显然一个 token 不够，故可以引入基于多模态双流注意力的MM-DiT：（扩散和自回归结合的模型，自回归上下文越来越长，也是运用了此方法）

先对图像流和文本流各自AdaLN（自适应层归一化）后各自计算Q、K、V，然后将图像的QKV和文本的QKV直接拼接（Concatenate）在一起，在这个拼接好的超长序列上，执行一次全局的自注意力机制，图像、文本可以内部信息融合，也可以让图像提取对应文本提示、文本根据图像更新上下文。

注意力计算完成后，将输出的超长序列重新切开，恢复成图像特征和文本特征。然后分别送入各自的残差连接和前馈神经网络（FFN），并应用AdaLN预测的门控系数。

（此处图片与上文 adaLN-Zero 初始化和 Unpatchification 说明重复，已合并转写。）

4. 注意力模块和自回归Transformer的不同点

对于扩散模型，在任何一个注意力层，所有的 token 必须同时相互计算注意力，最终也会同时输出所有 token 对应的向量。这就要求在显存中实打实地实例化一个 $N\times N$ 的注意力分数矩阵。而且如果扩散 $K$ 步，就要对这个矩阵计算 $K$ 次。对于视频生成模型，$N=\text{帧数}\times\text{patch数}$，这也就是为什么很多视频生成模型采用“块内扩散、块间自回归”的方式。

**训练阶段工作流**

1. **图像编码**：将原始图像 $x$ 输入冻结权重的 VAE 编码器，提取潜在特征 $z_0$。
2. **随机加噪**：随机采样一个时间步 $t\sim U(1,T)$ 和标准高斯噪声 $\epsilon\sim\mathcal{N}(0,I)$。根据前向公式计算得到加噪后的潜在变量 $z_t$。
3. **块化处理（Patchify）**：将 $z_t$ 切分为多个 Patch，展平并加上位置编码，形成输入序列。
4. **条件嵌入**：将时间步 $t$ 和图像对应的类别标签 $c$ 转化为条件向量（通过 Embedding 层和 MLP）。
5. **DiT 前向传播**：
   - 将 Token 序列输入 $N$ 层 DiT Block。
   - 在每一层内部，条件向量动态生成 adaLN 的缩放和平移参数，对 Token 进行自适应归一化。
6. **特征还原（Unpatchify）**：将最后一层的输出序列恢复为与 $z_t$ 相同的空间维度形状。
7. **损失计算与反向传播**：网络输出预测的噪声 $\epsilon_\theta$ 和协方差矩阵对角线元素。计算 $L_{\mathrm{simple}}+L_{\mathrm{vlb}}$，并通过梯度下降更新 Transformer 的权重。

**推理（生成）阶段工作流**

1. **噪声初始化**：从标准正态分布中采样一个纯噪声张量 $z_T\sim\mathcal{N}(0,I)$，并指定想要生成的类别标签 $c$。
2. **逐步去噪循环**：对于时间步 $t=T,T-1,\ldots,1$ 执行以下操作：
   - 将当前的 $z_t$ 切分为 Patch 序列。
   - 将序列连同时间步 $t$ 和类别 $c$ 送入 DiT 网络。
   - 网络通过 adaLN-Zero 机制处理上下文，预测出当前步骤的噪声 $\epsilon_\theta$ 和方差 $\Sigma_\theta$。
   - 使用预测结果，根据 DDPM 或 DDIM 采样算法的数学公式，计算出上一步的潜在变量 $z_{t-1}$。
3. **图像解码**：循环结束后得到去噪完毕的 $z_0$。将 $z_0$ 送入冻结权重的 VAE 解码器，重建出最终的高清像素图像 $\hat{x}$。

## 二、训练目标

在 DiT 的训练中，网络 $\theta$ 实际上需要同时预测加入的噪声 $\epsilon_\theta$ 以及方差 $\Sigma_\theta$。

为了预测噪声，使用简化的均方误差损失（Simplified Loss）：

$$
L_{\mathrm{simple}}=\mathbb{E}_{z_0,\epsilon,t}\left[\lVert \epsilon-\epsilon_\theta(z_t,t,c)\rVert_2^2\right]
$$

为了学习方差 $\Sigma_\theta$，模型额外引入了完整的变分下界（Variational Lower Bound, VLB）损失。总损失函数为两者的结合，通常对 $L_{\mathrm{vlb}}$ 施加一个较小的权重以防止干扰主目标的优化：

$$
L=L_{\mathrm{simple}}+L_{\mathrm{vlb}}
$$

## 三、方差的预测

### （一）为什么目标函数含预测方差项？

在逆向去噪过程 $p_\theta(z_{t-1}\mid z_t)$ 中，我们需要假设一个高斯分布：

$$
p_\theta(z_{t-1}\mid z_t)=\mathcal{N}(z_{t-1};\mu_\theta(z_t,t),\Sigma_\theta(z_t,t))
$$

在最初的 DDPM（Ho et al., 2020）中，作者发现真正的后验分布方差存在两个理论上的极端边界（上界和下界）：

1. **上界**：$\beta_t$（假设真实图像完全是标准正态噪声）。
2. **下界**：$\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$（假设真实图像 $z_0$ 是完全已知且确定的）。
具体解释：

既然模型只能看到一张充满噪点的图 $z_t$，那么从模型的视角来看，$z_0$ 就绝对不是一个确定的点，而是一个**充满无数种可能性的概率分布**。

我们可以构建一个直观的思想实验：

- **极端情况 A（$t$ 很大，接近纯噪声）**：假设 $z_t$ 就像是一台完全没有信号的雪花屏电视。模型看着这片雪花，它能确定原本的画面是一只猫、一只狗，还是一辆车吗？完全不能。因为有成千上万种不同的 $z_0$ 在加上大量噪声后，都会坍缩成这同一片雪花。此时，模型对真实的 $z_0$ 极度不确定，这导致它推断上一步图像的方差极大，逼近理论上界 $\beta_t$。
- **极端情况 B（$t$ 很小，接近去噪尾声）**：假设 $z_t$ 是一张非常清晰的猫咪照片，只是上面落了几粒灰尘（微小噪声）。模型看着这张图，非常笃定这原本就是一只猫（此时其他 $z_0$ 的可能性已经极小）。此时，模型对 $z_0$ 非常确定，逆向去噪的方差极小，逼近理论下界 $\tilde{\beta}_t$。
在我们训练的时候，我们知道真实图像就在训练集里，完全已知且确定，但模型不知道；模型在推理的时候，若对真实的 $z_0$ 是什么完全没有把握，则 $z_{t-1}$ 的方差取上界；若对真实的 $z_0$ 是什么完全确定，则 $z_{t-1}$ 的方差取下界。

上界为什么为 $\beta_t$：

在逆向过程中，我们想要推导的是 $q(z_{t-1}\mid z_t)$。根据贝叶斯公式，我们可以将其展开为：

$$
q(z_{t-1}\mid z_t)=\frac{q(z_t\mid z_{t-1})q(z_{t-1})}{q(z_t)}
$$

- $q(z_t\mid z_{t-1})$ 是前向加噪过程，这是完全已知的：$\mathcal{N}(z_t;\sqrt{\alpha_t}z_{t-1},\beta_t I)$。
- **关键点来了**：在不确定性最大的极端情况（通常在 $t$ 很大、接近扩散末期时），图像已经被破坏得面目全非。此时如果我们假设完全不知道 $z_0$ 是什么，那么对于 $z_{t-1}$ 边缘分布的最合理假设，就是它已经退化成了一个**标准正态分布（先验分布）**：

$$
q(z_{t-1})\approx\mathcal{N}(0,I)
$$

于是有：

$$
q(z_{t-1}\mid z_t)\propto q(z_t\mid z_{t-1})q(z_{t-1})
$$

$$
q(z_{t-1}\mid z_t)\propto \exp\left(-\frac{1}{2}\left[\frac{(z_t-\sqrt{\alpha_t}z_{t-1})^2}{\beta_t}+z_{t-1}^2\right]\right)
$$

配方后，根据正态分布的系数可确定方差。

初代 DDPM 认为，只要总的扩散步数 $T$ 足够大（例如 $T=1000$），每一步的变化极小，此时 $\beta_t$ 和 $\tilde{\beta}_t$ 几乎相等。因此，**初代模型选择不预测方差**，而是直接将方差暴力固定为一个常数矩阵：

$$
\Sigma_\theta(z_t,t)=\sigma_t^2 I
$$

其中 $\sigma_t^2$ 被硬编码为 $\beta_t$ 或 $\tilde{\beta}_t$。

**致命痛点：**

这种做法在 $T=1000$ 时效果很好，但如果我们想**加速采样**，比如只用 50 步（大步长采样），这两个方差边界就会产生巨大的分歧。此时如果仍然使用固定的方差，模型就会被强制拉入次优的概率分布，导致生成的图像充满噪点或极其模糊。

### （二）如何预测方差？

为了解决大步长采样带来的方差不确定性，Improved DDPM 提出（并被 DiT 采用），让神经网络自己根据当前的特征 $z_t$ 和时间步 $t$ 来动态预测最合适的方差。

1. **方差的参数化（Parameterization）**

直接让模型输出绝对的方差数值是不稳定的。因此，模型被设计为输出一个插值系数向量 $v\in[0,1]^d$，在对数空间内对理论的上下界进行线性插值：

$$
\log\Sigma_\theta(z_t,t)=v\odot\log\beta_t+(1-v)\odot\log\tilde{\beta}_t
$$

注：这也是为什么在 DiT 的最后一个全连接层，特征维度会扩展到 $2c$。其中 $c$ 个通道用于预测噪声 $\epsilon_\theta$，另外 $c$ 个通道正是用来预测这个插值向量 $v$。

2. **梯度的阻断与损失函数（Stop-Gradient and Loss）**

为了训练这个动态方差，我们必须引入完整的变分下界（Variational Lower Bound, VLB）损失：

$$
L_{\mathrm{vlb}}=\sum_{t=1}^{T}D_{\mathrm{KL}}\left(q(z_{t-1}\mid z_t,z_0)\parallel p_\theta(z_{t-1}\mid z_t)\right)
$$

由于 $L_{\mathrm{vlb}}$ 在计算时同时包含了均值 $\mu_\theta$ 和方差 $\Sigma_\theta$，直接优化它会导致均值预测的梯度变得极其不稳定，反而破坏图像质量。

因此，DiT 在计算 $L_{\mathrm{vlb}}$ 时，对均值预测施加了**停止梯度（Stop-Gradient）**操作。这意味着 $L_{\mathrm{vlb}}$ 的反向传播梯度只用来更新方差 $v$ 的权重，而完全不干扰噪声 $\epsilon_\theta$ 的学习。

最终的混合损失函数表示为：

$$
L=L_{\mathrm{simple}}+\lambda L_{\mathrm{vlb}}
$$

在 DiT 中，通常设置 $\lambda=0.001$，以确保方差学习不会干扰主体的噪声预测。

## 参考文献

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR.
- Peebles, W., & Xie, S. (2023). [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). ICCV.
