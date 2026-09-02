---
title: "26.7 扩散模型中的强化学习与DPO for Diffusion"
source_docx: "第5部分 扩散模型与多模态生成/26.扩散模型/26.7 扩散模型中的强化学习与DPO for Diffusion.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 26.7 扩散模型中的强化学习与DPO for Diffusion

## 一、扩散模型中 RL 的困境

### 1. 稀疏奖励下的长轨迹信度分配难题（Credit Assignment over Long Trajectories）

- **困境所在**：RL 的奖励信号，如美学评分、人类偏好对齐评分 $R(x_0)$，极其稀疏，只能在生成过程结束、得到最终完整图像后才能获取。由于中间没有任何即时奖励（Dense Reward），一旦最终图像评分很低，RL 算法极难搞清楚到底是哪一步，例如 $t=0.9$ 的大轮廓生成阶段，还是 $t=0.1$ 的细节纹理阶段，的神经网络输出出现了失误。
- **后果**：传统的优势函数估计，如 GAE，在这种长链条的 ODE/SDE 求解器中难以向回精准传播奖励梯度，导致优化极其低效，甚至出现错误的信用归因。

尤其是价值网络训练极易崩溃，这也就是基于 GRPO 的算法更常用的原因。

### 2. 超高维连续动作空间的探索危机（Exploration in High-Dimensional Continuous Space）

- **困境所在**：要在如此庞大的连续高维空间中进行有效的随机探索（Exploration），无异于大海捞针。如果在每一步预测的向量场或噪声上盲目叠加高斯探索噪声，极易导致整个微分方程的积分轨迹出现级联误差（Cascading Errors），使最终生成的图像彻底崩溃成无意义的噪点。
- **后果**：为了保证生成轨迹不崩溃，研究者只能将探索噪声的方差设置得极小，但这又会导致模型缺乏探索能力，迅速陷入局部最优。这也是 DM+RL 极易引发模式崩溃（Mode Collapse）、丧失生成多样性的底层原因。

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
\log p_\theta(x\mid c)
&\approx -\mathbb{E}_{t,\epsilon}\left[\mathcal{L}_\theta(x,c,t)\right]+C
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

在后面会提到的流匹配模型中同理：

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
5. 反向传播更新：将上述四个误差标量代入 $\mathcal{L}_{\mathrm{DPO}\text{-}\mathrm{Diff}}(\theta)$ 公式中。通过深度学习框架（如 PyTorch）的自动求导机制计算关于 $\theta$ 的梯度，并使用优化器（如 AdamW）更新模型权重。

## 参考文献

- Black, K., Janner, M., Du, Y., Kostrikov, I., & Levine, S. (2023). [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301). arXiv:2305.13301.
- Wallace, B., Dang, M., Rafailov, R., et al. (2024). [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908). CVPR 2024.
- Liu, J., Liu, G., Liang, J., et al. (2025). [Improving Video Generation with Human Feedback](https://arxiv.org/abs/2501.13918). NeurIPS.
