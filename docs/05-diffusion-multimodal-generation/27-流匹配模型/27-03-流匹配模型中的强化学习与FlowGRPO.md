---
title: "27.3 流匹配模型中的强化学习与FlowGRPO"
source_docx: "第5部分 扩散模型与多模态生成/27.流匹配模型/27.3 流匹配模型中的强化学习与FlowGRPO.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 27.3 流匹配模型中的强化学习与FlowGRPO

## 一、马尔可夫过程的变化

在流匹配模型下，马尔可夫过程的形式和纯自回归的 LLM 有所不同：

1. **状态 $x_t$**：当前时间步的连续图像或潜变量。
2. **动作 $v_t$**：模型输出的连续向量场，即一个方向。
3. **环境转移**：确定性的数值积分，例如欧拉法 $x_{t+\Delta t}=x_t+v_t\Delta t$。
4. **奖励**：只有积分到 $t=1$ 得到最终图像 $x_1$ 后，才能获得美学评分或对齐评分 $R(x_1)$。

从 $x_0$ 到 $x_1$ 的过程，实质上是 $v_t$ 关于 $t$ 从 0 到 1 积分，这里 $v_t$ 是一个连续函数。事实上一般是采样几十步，每次沿该时间步的速度走一小步。

## 二、流匹配中 RL 的特殊难点

前面提到过扩散模型中运用 RL 存在的两个难点：稀疏奖励下的长轨迹信度分配，以及超高维连续动作空间的探索危机。此外，流匹配模型受限于自身底层数学假设，在 RL 中还有一些独有的难题。

### 1. 破坏最优传输的“直线特性”与流形脱离（Breaking Optimal Transport Straight Lines）

- **困境所在**：流匹配，尤其是目前主流的 OT-FM，在预训练时的最大卖点，是通过最优传输（Optimal Transport）强制对齐纯噪声 $x_1$ 和真实图像 $x_0$，构造了一条极其简单的零曲率直线轨迹：

$$
x_t=t x_1+(1-t)x_0
$$

- **直线破裂**：预训练的 FM 是一个完美遵循这条直线的拟合器。当引入 RL 时，RL 策略梯度的唯一目标是“最大化最终图像的奖励”。为了迎合奖励，RL 会强行修改中间时间步的 $v_t$，这就不可避免地将 ODE 轨迹“拽离”了原本的最优传输直线。
- **后果**：扩散模型原本的轨迹就是弯曲的高曲率布朗运动，所以稍微拽偏一点尚能承受。但对于 FM 而言，一旦轨迹脱离直线，不仅会产生偏离流形、生成无意义图像的现象，更致命的是，轨迹曲率的激增会导致 ODE 求解器的截断误差（Truncation Error）爆炸。原本 FM 只需要 4 步积分就能出图，被 RL 扭曲后，必须增加到几十甚至上百步才能维持图像质量，直接粉碎了 FM 最大的“少步数采样优势”。

### 2. 确定性 ODE 带来的“似然度计算死锁”（Likelihood Degeneration）

- **困境所在**：标准在线 RL，如 PPO/GRPO，必须计算动作的对数似然度 $\log\pi(a\mid s)$。DM 的逆向去噪本身就是随机高斯转移，天然自带可计算的概率密度。
- **数学死锁**：FM 的生成过程是纯确定性的常微分方程（ODE）。这意味着给定状态 $x_t$，其转移到下一步 $x_{t-\Delta t}$ 的概率在数学上是一个狄拉克 $\delta$ 函数（Dirac delta function）。对 $\delta$ 函数求 log 会导致无穷大（Degenerate Likelihood）。因此，标准 RL 算法在纯粹的流匹配上是完全无法运行的。必须强行魔改底层架构，例如通过 ODE-to-SDE 转换人为注入探索噪声，这让原本极其优雅简洁的 FM 变得异常臃肿和不稳定。

当然需要注意的是，流匹配模型在给定输入的情况下输出是固定的，不代表其输出不存在概率分布。在流匹配中，我们常说“去噪过程是固定的”，指的是给定一个具体的初始噪声 $z$ 后，它演化到目标数据 $x_1$ 的路径（Trajectory）是唯一且确定的，因为求解的是常微分方程 ODE，而非随机微分方程 SDE。

但是，模型具有分布，是因为输入源头是一个分布。这在数学上被称为前推测度（Push-forward Measure）。可以把标准高斯分布的噪声 $z\sim\mathcal{N}(0,I)$ 想象成一把沙子，并把沙子撒在一个有特定风向，即向量场 $v_t(x)$，的传送带上。虽然每一粒沙子被风吹动的轨迹是完全确定、固定的，但一开始沙子落在各个位置的概率不同。当它们最终被吹到终点时，就会自然而然地堆积成一个新的形状，这就是模型生成的数据分布 $p_1(x)$。

## 三、FlowGRPO：化 ODE 为 SDE

### （一）ODE 转化为 SDE 的方法

- 背景原理：在标准的流匹配（Flow Matching）或扩散模型中，图像生成被建模为一个确定性的 ODE。这意味着，只要给定一个初始纯噪声，模型去噪生成图像的轨迹就是唯一确定、不可改变的。
- RL的困境与解法：强化学习（RL）的本质是探索与利用（Exploration and Exploitation）。如果轨迹是确定的，模型就无法尝试新的生成路径，自然也就无从得知哪种路径能获得更高的奖励。因此，FlowGRPO 将这个确定的 ODE 转化为了 SDE。通过在每一步去噪过程中引入由 $g(t)$ 控制的随机噪声扰动，算法赋予了模型偏离既定路线去“试错”的能力。有了这种随机探索性，RL 才能发挥作用。

具体来说，FlowGRPO将流匹配中的常微分方程（ODE）：

$$
dx_t=v_\theta(x_t,t)dt
$$

转化为了随机微分方程：

$$
dx_t=\left[v_\theta(x_t,t)+g(t)\nabla\log p_t(x_t)\right]dt+\sqrt{2g(t)}\,dw_t
$$

仔细观察，可以发现加入的项来自扩散随机加噪的随机梯度朗之万动力学表达式：

$$
dx_t=\nabla\log p(x_t)dt+\sqrt{2}\,dw_t
$$

其中第二项为探索噪声，第一项为保证 $x$ 关于 $t$ 的边缘分布不变的修正项，避免模型遇到OOD情形。

乘的系数 $g(t)$ 表示布朗运动的剧烈程度，是一个预定义好的函数。

### （二）比例归一化与截断机制的修复

- 背景原理：在 PPO 或 GRPO 等强化学习算法中，为了保证训练的稳定性，必须使用“截断机制”（Clipping）。它通过计算新旧策略的“重要性比值”（Importance Ratio），强制限制模型每次参数更新的幅度，防止模型因为某次偶然的高奖励而彻底破坏原本的生成能力。

在流匹配的 SDE 采样中，每一步的去噪转移概率是一个高斯分布。设定当前步的时间间隔为 $\Delta t$，噪声方差为 $\sigma_{t_k}^2\Delta t$。新策略（当前正在优化的模型）和旧策略（收集数据时的参考模型）的概率密度函数都可以写成高斯形式。重要性比值定义为新旧策略概率之比 $r_{t_k}(\theta)$，取自然对数后为：

$$
\begin{aligned}
\log r_{t_k}(\theta)
&=
-\frac{1}{2\sigma_{t_k}^2\Delta t}
\left\|x_{t_k-\Delta t}-\mu_\theta\right\|^2
\\
&\quad+
\frac{1}{2\sigma_{t_k}^2\Delta t}
\left\|x_{t_k-\Delta t}-\mu_{\theta_{old}}\right\|^2
\end{aligned}
$$

关键点来了：在强化学习中，训练数据的轨迹 $x_{t_k-\Delta t}$ 是由旧策略 $p_{\theta_{old}}$ 采样生成的。因此它必然满足：

$$
\begin{aligned}
x_{t_k-\Delta t}
&=
\mu_{\theta_{old}}+\sqrt{\sigma_{t_k}^2\Delta t}\cdot\epsilon
\end{aligned}
$$

这里 $\epsilon\sim\mathcal{N}(0,I)$ 是标准高斯噪声。将这个 $x_{t_k-\Delta t}$ 代入对数公式，并定义新旧策略的均值差为 $\Delta\mu=\mu_{\theta_{old}}-\mu_\theta$，公式展开并化简后得到：

$$
\begin{aligned}
\log r_{t_k}(\theta)
&=
-\frac{\|\Delta\mu\|^2}{2\sigma_{t_k}^2\Delta t}
\\
&\quad-
\frac{\epsilon^T\Delta\mu}{\sigma_{t_k}\sqrt{\Delta t}}
\end{aligned}
$$

对上述公式求高斯噪声 $\epsilon$ 的数学期望，由于 $\mathbb{E}[\epsilon]=0$，第二项被消去：

$$
\begin{aligned}
\mathbb{E}_{\epsilon}\left[\log r_{t_k}(\theta)\right]
&=
-\frac{\|\Delta\mu\|^2}{2\sigma_{t_k}^2\Delta t}
\end{aligned}
$$

因为 $\|\Delta\mu\|^2>0$（只要策略更新了，均值就会有差异），所以对数重要性比值的期望严格小于 0。

- 截断失效的危机：在流匹配模型中，由于概率密度的特性，这个重要性比值的分布会系统性地向左偏移（平均值小于 1），并且在不同的时间步长上表现出极不一致的方差。这导致预设的截断边界（比如 $[1-\epsilon,1+\epsilon]$）形同虚设，无法有效约束模型那些过于自信的错误更新，进而引发严重的“奖励黑客”（Reward Hacking）现象，即模型钻空子获得高分但生成一堆无意义的乱象。

- RatioNorm 的解法：为了修复这个问题，论文引入了 RatioNorm 技术。该技术对数重要性比值进行标准化处理，强行将其分布重新居中到零附近。这样一来，截断边界就能再次精准地“卡”住过大的策略更新，确保了最终导出的图像目标函数 $\mathcal{J}_{Flow}(\theta)$ 能够平稳收敛。

$$
\begin{aligned}
\log \tilde{r}_{t_k}(\theta)
&=
\sigma_{t_k}\sqrt{\Delta t}
\left(
\log r_{t_k}(\theta)
\right.
\\
&\quad\left.
+\frac{\|\Delta\mu_\theta(x_{t_k},t_k)\|^2}{2\sigma_{t_k}^2\Delta t}
\right)
\end{aligned}
$$

## 参考文献

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR.
- Black, K., Janner, M., Du, Y., Kostrikov, I., & Levine, S. (2023). [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301). arXiv:2305.13301.
- Liu, J., Liu, G., Liang, J., et al. (2025). [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470). arXiv:2505.05470.
