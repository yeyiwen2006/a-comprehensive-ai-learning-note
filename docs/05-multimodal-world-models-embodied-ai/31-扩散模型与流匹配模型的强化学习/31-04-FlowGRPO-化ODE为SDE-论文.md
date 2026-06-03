---
title: "31.4 FlowGRPO：化ODE为SDE（论文）"
source_docx: "第5部分 多模态生成、世界模型与具身智能/31.扩散模型与流匹配模型的强化学习/31.4 FlowGRPO：化ODE为SDE（论文）.docx"
status: "auto-converted"
ocr: "image placeholders rebuilt as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 31.4 FlowGRPO：化ODE为SDE（论文）

> 本文是论文阅读笔记，内容代表对应论文方法或作者理解，不应直接视为领域共识或工程最佳实践。

## 一、从ODE到SDE的变换

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

## 二、比例归一化与截断机制的修复

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

- Liu, J., Liu, G., Liang, J., et al. (2025). [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470). arXiv:2505.05470.
- Shao, Z., Wang, P., Zhu, Q., et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv:2402.03300.
