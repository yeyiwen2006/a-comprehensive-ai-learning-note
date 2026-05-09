---
title: "31.2 Offline RL转化为加权监督学习（论文）"
source_docx: "第5部分 世界模型、多模态生成与具身智能/31.扩散模型与流匹配模型的强化学习/31.2 Offline RL转化为加权监督学习（论文）.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 31.2 Offline RL转化为加权监督学习（论文）


> 本文是论文阅读笔记，内容代表对应论文方法或作者理解，不应直接视为领域共识或工程最佳实践。

## 一、RLVR转化为逼近最优策略的加权监督学习

### （一）核心思想

该方法的核心思想是 **“避开在线探索的难题，将 RL 转化为加权的监督学习”**。

- 它首先定义了一个代理散度 $D_{FM}$，试图让当前模型 $\pi_\theta$ 在流匹配损失空间内逼近理论最优策略 $\pi^{*}$。
- 由于无法直接从 $\pi^{*}$ 采样，它利用重要性采样（Importance Sampling），引入静态离线数据集 $D$，并赋予权重 $w(o, a) \propto \exp(A / \beta)$，即优势 $A$ 越大的动作，权重越高。
- 点睛之笔在于最后的“极端设定”：通过让折扣因子 $\gamma \to 1$ 并给予极端负奖励，优势函数被极化，权重 $w(o, a)$ 变成了二值 $\{0, 1\}$。这在数学上等价于直接丢弃所有“失败”的轨迹，只保留“成功”的轨迹进行监督学习。

### （二）数学表达

在标准的正则化强化学习框架下，我们的目标是在最大化奖励的同时，约束新策略 $\pi_\theta$ 不要偏离参考策略 $\pi_{ref}$ 太远。其目标函数定义为：

$$
\begin{aligned}
J(\theta)
&= \mathbb{E}_{\tau \sim p_\theta}
\left[
R(\tau)
- \beta \mathbb{E}_{o \sim p_\theta}
\left[D(\pi_\theta(\cdot \mid o) \Vert \pi_{ref}(\cdot \mid o))\right]
\right]
\end{aligned}
$$

根据数学推导，这个目标函数存在一个理论上的最优策略闭式解 $\pi^{*}$：

$$
\begin{aligned}
\pi^{*}(a \mid o)
&\propto
\pi_{ref}(a \mid o)
\exp\left(\frac{A^{\pi_{ref}}(o, a)}{\beta}\right)
\end{aligned}
$$

这意味着，最优策略倾向于在参考策略的基础上，给予那些具有高优势函数（Advantage Function，$A > 0$）的动作更高的概率。

因为这个理论最优策略 $\pi^{*}$ 通常极其复杂，无法直接用一个有限参数的神经网络表示，所以我们需要进行策略投影（Policy Projection）：即找一个参数化的策略 $\pi_\theta$，让它尽可能地去拟合 $\pi^{*}$。

在传统的优势加权回归（AWR）算法中，拟合的手段是最小化 $\pi^{*}$ 和 $\pi_\theta$ 之间的 KL 散度：

$$
\begin{aligned}
\theta^{*}
&= \arg\min_\theta
\mathbb{E}_{o \sim D}
\left[
D_{\mathrm{KL}}(\pi^{*}(\cdot \mid o) \Vert \pi_\theta(\cdot \mid o))
\right]
\end{aligned}
$$

展开 KL 散度后，这等价于最大化加权对数似然（Weighted Log-likelihood）：$\mathbb{E}[w(o, a)\log \pi_\theta(a \mid o)]$。

**问题就在这里**：我们的 VLA 模型是用流匹配目标 $\mathcal{L}_{FM}$ 训练的，它根本算不出确切的 $\log \pi_\theta(a \mid o)$。这条传统路线走不通。
替代散度法：

$$
\begin{aligned}
D_{FM}(\pi^{*}(\cdot \mid o), \pi_\theta(\cdot \mid o))
&\triangleq
\mathbb{E}_{a \sim \pi^{*}(\cdot \mid o)}
\left[\mathcal{L}_{FM}(\theta; o, a)\right]
\end{aligned}
$$

通俗解释：这个公式的物理意义是，“假设我现在能从理论最优策略 $\pi^{*}$ 中抽取出完美的动作样本 $a$，那么我就用模型自带的流匹配损失函数 $\mathcal{L}_{FM}$，强迫我的神经网络 $\pi_\theta$ 去学习、生成这些完美的动作。”

这就巧妙地避开了计算概率分布的要求，直接在流匹配的损失空间里衡量两个策略的“距离”。

有了新的散度 $D_{FM}$，策略投影步骤就变成了：

$$
\begin{aligned}
\theta^{*}
&= \arg\min_\theta
\mathbb{E}_{o \sim D}
\mathbb{E}_{a \sim \pi^{*}(\cdot \mid o)}
\left[
\mathcal{L}_{FM}(\theta; o, a)
\right]
\end{aligned}
$$

但我们在现实中显然无法真的从未知的理论最优策略 $\pi^{*}$ 中采样。因此，论文采用了离线强化学习中标准的“重要性采样（Importance Sampling）”技巧：我们用手头收集到的固定数据集 $D$ 中的样本来代替 $\pi^{*}$ 中的样本，并给这些样本乘以一个权重 $w(o, a)$，权重正比于指数化的优势函数 $\exp(A / \beta)$。

推导到这一步，目标函数变为了：

$$
\begin{aligned}
\theta^{*}
&\approx \arg\min_\theta
\mathbb{E}_{(o,a) \sim D}
\left[
w(o, a)\mathcal{L}_{FM}(\theta; o, a)
\right]
\end{aligned}
$$

最后，为了极致地简化和易于扩展，论文做了一个极端设定：将折扣因子 $\gamma$ 设为接近 $1$，并给失败的轨迹分配极大的负奖励。这导致权重 $w(o, a)$ 退化成了二值权重（成功的动作权重为 $1$，失败的为 $0$）。这就是论文中方程（4）最终用于策略微调的监督学习目标函数的由来。

## 二、DPO for Diffusion

### （一）数学表达

第一步：RLHF 目标的重参数化

在标准的带 KL 惩罚的 RL 框架下，我们希望微调一个策略（扩散模型）$p_\theta(x\mid c)$，使其在给定条件 prompt $c$ 时，生成的图像 $x$ 能最大化隐含的奖励 $r(x,c)$，同时不能偏离预训练的参考模型 $p_{\mathrm{ref}}(x\mid c)$ 太远：

$$
\begin{aligned}
\max_\theta\ \mathbb{E}_{x\sim p_\theta}\left[r(x,c)\right]
&\quad-\beta D_{\mathrm{KL}}\left(p_\theta(x\mid c)\Vert p_{\mathrm{ref}}(x\mid c)\right)
\end{aligned}
$$

其中 $\beta$ 是控制偏离程度的超参数。这个优化问题存在一个理论上的闭式最优解（Optimal Policy）：

$$
p^{*}(x\mid c)=\frac{1}{Z}p_{\mathrm{ref}}(x\mid c)\exp\left(\frac{1}{\beta}r(x,c)\right)
$$

其中 $Z$ 是配分函数。DPO 的核心做法在于对上式进行移项，反解出奖励函数 $r(x,c)$：

$$
r(x,c)=\beta\log\frac{p^{*}(x\mid c)}{p_{\mathrm{ref}}(x\mid c)}+\beta\log Z
$$

第二步：代入 Bradley-Terry 偏好模型

在偏好学习中，我们通常假设人类偏好服从 Bradley-Terry（BT）模型。即对于给定的 prompt $c$，人类认为图像 $x_w$（winner，获胜者）比 $x_l$（loser，失败者）更好的概率为：

$$
p(x_w\succ x_l\mid c)=\sigma\left(r(x_w,c)-r(x_l,c)\right)
$$

其中 $\sigma$ 是 Sigmoid 函数。将第一步反解出的 $r(x,c)$ 代入 BT 模型中，由于是计算差值，常数项 $\beta\log Z$ 被消去：

$$
\begin{aligned}
p(x_w\succ x_l\mid c)
&=\sigma\left(
\beta\log\frac{p_\theta(x_w\mid c)}{p_{\mathrm{ref}}(x_w\mid c)}
-\beta\log\frac{p_\theta(x_l\mid c)}{p_{\mathrm{ref}}(x_l\mid c)}
\right)
\end{aligned}
$$

这就是大语言模型中标准 DPO 的目标函数。但扩散模型会遇到一个关键问题：扩散模型的精确对数似然 $\log p_\theta(x\mid c)$ 在数学上很难计算。

第三步：扩散模型的 ELBO 替换

Diffusion-DPO 的最大贡献，就是用证据下界（ELBO，Evidence Lower Bound）来近似这个难以计算的对数似然。在扩散模型中，极大化对数似然等价于极小化去噪预测误差（MSE 损失）。令 $\mathcal{L}_\theta(x,c,t)$ 为模型在时间步 $t$ 的去噪损失（以预测噪声 $\epsilon$ 为例）：

$$
\begin{aligned}
\mathcal{L}_\theta(x,c,t)
&=\left\lVert \epsilon_\theta(x_t,c,t)-\epsilon\right\rVert_2^2
\end{aligned}
$$

因为似然与去噪损失成负相关，我们可以做如下近似：

$$
\begin{aligned}
\log p_\theta(x\mid c)\approx
&-\mathbb{E}_{t,\epsilon}\left[\mathcal{L}_\theta(x,c,t)\right]+C
\end{aligned}
$$

因此，对数似然的比值可以替换为去噪损失的差值：

$$
\begin{aligned}
\log\frac{p_\theta(x\mid c)}{p_{\mathrm{ref}}(x\mid c)}
&\approx
\mathbb{E}_{t,\epsilon}
\left[
\mathcal{L}_{\mathrm{ref}}(x,c,t)-\mathcal{L}_\theta(x,c,t)
\right]
\end{aligned}
$$

这个公式的物理直觉非常清晰：如果参考模型和当前模型在“好图” $x_w$ 上的损失差，大于它们在“坏图” $x_l$ 上的损失差，我们就施加奖励；反之则施加惩罚。

在流匹配中同理：

在流匹配中，对数似然依然难以精确计算，但我们可以用向量场拟合的 MSE 损失来替代扩散模型中的噪声预测损失。只需将公式中的 $\mathcal{L}_\theta$ 替换为我们在 DMD 中讨论过的流匹配目标：

$$
\begin{aligned}
\mathcal{L}^{\mathrm{FM}}_\theta(x,c,t)
&=\left\lVert v_\theta(x_t,c,t)-(x_1-x_0)\right\rVert_2^2
\end{aligned}
$$

只要将这个向量场误差代入 Diffusion-DPO 的框架中（近期文献中称为 Flow-DPO），就可以利用离线偏好对来直接微调流匹配模型，而不需要涉及在线 RL 求解 ODE 的复杂过程。

### （二）工作流

阶段 1：数据准备

1. 构建或收集一个静态离线偏好数据集 $\mathcal{D}$。数据集中包含大量的元组 $(c,x_w,x_l)$，即“文本提示词”、“人类或 AI 偏好的好图”、“人类或 AI 淘汰的坏图”。
2. 准备预训练好的基础扩散模型作为参考模型 $\theta_{\mathrm{ref}}$，并将其权重冻结。
3. 初始化一个架构完全相同的策略模型 $\theta$，即要训练的模型，通常从 $\theta_{\mathrm{ref}}$ 初始化。

阶段 2：训练循环（Batch 级别）

对于每一个训练 step，执行以下操作：

1. 采样与加噪：从数据集中采样一个 batch 的 $(c,x_w,x_l)$。随机采样时间步 $t\sim\mathcal{U}(0,T)$，并采样高斯噪声 $\epsilon\sim\mathcal{N}(0,I)$。
2. 构建中间状态：利用前向扩散过程的公式，分别给好图和坏图加噪到时间步 $t$：

$$
\begin{aligned}
x_{w,t} &= \sqrt{\bar{\alpha}_t}x_w+\sqrt{1-\bar{\alpha}_t}\epsilon,\\
x_{l,t} &= \sqrt{\bar{\alpha}_t}x_l+\sqrt{1-\bar{\alpha}_t}\epsilon.
\end{aligned}
$$

3. 计算参考模型的误差（No Gradient）：将加噪后的好图、坏图连同 $t$ 和 $c$ 输入冻结的参考模型 $\theta_{\mathrm{ref}}$，计算它们预测噪声与真实噪声 $\epsilon$ 的均方误差：$\mathcal{L}_{\mathrm{ref}}(x_w)$ 和 $\mathcal{L}_{\mathrm{ref}}(x_l)$。
4. 计算策略模型的误差（With Gradient）：将同样的数据输入正在训练的策略模型 $\theta$，计算其预测误差：$\mathcal{L}_\theta(x_w)$ 和 $\mathcal{L}_\theta(x_l)$。
5. 反向传播更新：将上述四个误差标量代入 $\mathcal{L}_{\mathrm{DPO-Diff}}(\theta)$ 公式中。通过深度学习框架（如 PyTorch）的自动求导机制计算关于 $\theta$ 的梯度，并使用优化器（如 AdamW）更新模型权重。

## 参考文献

- Peters, J., & Schaal, S. (2007). [Reinforcement Learning by Reward-Weighted Regression for Operational Space Control](https://dl.acm.org/doi/10.1145/1273496.1273590). ICML.
- Peng, X. B., Kumar, A., Zhang, G., & Levine, S. (2019). [Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning](https://arxiv.org/abs/1910.00177). arXiv:1910.00177.
- Nair, A., Dalal, M., Gupta, A., & Levine, S. (2020). [Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359). arXiv:2006.09359.
- Wallace, B., Gokul, A., Ermon, S., & Naik, N. (2024). [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908). CVPR.
