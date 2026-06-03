---
title: "31.3 每步引入可学习噪声的Online RL（论文）"
source_docx: "第5部分 多模态生成、世界模型与具身智能/31.扩散模型与流匹配模型的强化学习/31.3 每步引入可学习噪声的Online RL（论文）.docx"
status: "auto-converted"
ocr: "image placeholders rebuilt as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 31.3 每步引入可学习噪声的Online RL（论文）

> 本文是论文阅读笔记，内容代表对应论文方法或作者理解，不应直接视为领域共识或工程最佳实践。

Flow-Noise等方法在原有流匹配模型的基础上引入可学习的噪声网络，将去噪过程重塑为离散时间的MDP，从而可用传统RL算法，既解决了流匹配缺失随机性、无法计算动作概率似然的问题，又通过可学习、有方向的探索解决了高维空间盲目探索往往失败的问题。

## 一、核心思想

传统的流匹配是在连续时间内求解常微分方程（ODE），缺乏随机性，难以像 PPO 那样进行对数似然（log-likelihood）的计算和在线试错探索。Flow-Noise 的创新在于，它在原本确定性的去噪路径中，主动注入了一个可学习的噪声网络，将去噪过程强行重塑为一个离散时间的马尔可夫决策过程（Discrete-time MDP）。这使得模型在每一步生成时既有探索能力，又能获得精确的动作对数似然估算，从而适配传统的 Actor-Critic 在线 RL 算法。

## 二、算法原理

它放弃了纯确定性的 ODE 采样，而是引入了一个可学习的噪声网络（Learnable Noise Network），将原本连续的去噪轨迹离散化为若干步（步长为 $\delta$），并在每一步中注入受控的随机性。

在该内部去噪 MDP 中：

- 状态空间（State）：在去噪时间步 $\tau$，内部状态定义为当前环境观测与当前隐变量的组合：

$$
\bar{s}_t^\tau=(\mathbf{o}_t,\mathbf{A}_t^\tau)
$$

- 动作空间（Action）：内部的“动作”即为流模型生成的下一个离散时间步的隐状态（当 $\tau<1$ 时）或最终在环境中执行的物理动作（当 $\tau=1$ 时）：

$$
\bar{a}_t^\tau=
\begin{cases}
\mathbf{A}_t^{\tau+\delta}, & \tau<1,\\
\mathbf{A}_t^1, & \tau=1.
\end{cases}
$$

- 随机转移与可学习噪声：从 $\mathbf{A}_t^\tau$ 到 $\mathbf{A}_t^{\tau+\delta}$ 的转移被建模为高斯分布：

$$
\mathbf{A}_t^{\tau+\delta}=\mu_\tau+\sigma_\tau\delta\cdot\epsilon
$$

其中，$\epsilon\sim\mathcal{N}(0,I)$ 是注入的标准高斯噪声，$\mu_\tau$ 由流匹配网络 $v_\theta$ 预测出的均值方向决定，$\sigma_\tau$ 则是由可学习的噪声网络输出的标准差。

可学习噪声网络的关键作用包括：

- 自适应探索：在生成轨迹的安全区域（例如刚开始去噪、只有大轮廓时），网络会输出较大的方差以鼓励广泛探索；当进入对噪声极其敏感的高频细节生成区域时，会自动收敛输出极小的方差，甚至趋近于 0。
- 维度解耦：网络可以学习到在不重要的背景特征维度上加大探索，而在关键的语义维度上保持克制，从而避开高维空间中的级联崩溃。

由于每一步 $\tau\rightarrow\tau+\delta$ 都服从已知的高斯分布，整个去噪轨迹的联合概率密度就可以被精确分解并计算。给定一条从 $\tau=0$ 到 $\tau=1$ 的离散去噪轨迹，其总动作对数似然为每一步转移概率对数的累加：

$$
\begin{aligned}
\log \pi_\theta(\mathbf{a}_t \mid \mathbf{o}_t)
&=
\sum_{\tau=0}^{1-\delta}
\log p_\theta(\mathbf{A}_t^{\tau+\delta}\mid \mathbf{A}_t^\tau,\mathbf{o}_t)
\end{aligned}
$$

因为 $p_\theta$ 是由参数化的高斯分布构成，上式变得完全可计算，从而可以直接用于 PPO 等基于策略梯度的 RL 算法的目标函数中。

## 三、工作流

步骤 1：环境交互与初始化（外部循环起点）

1. 智能体从仿真环境或现实世界获取当前时间步的观测状态，如摄像头图像、本体感受数据等。
2. 在模型内部，初始化流匹配的起点，采样纯高斯噪声 $\mathbf{A}_t^0\sim\mathcal{N}(0,I)$。

步骤 2：离散化去噪过程（内部 MDP 循环）

开启一个从 $\tau=0$ 到 $\tau=1$（步长为 $\delta$）的离散去噪循环。对于每一步 $\tau$：

1. 构建内部状态：组合观测与隐变量得到 $\bar{s}_t^\tau$。
2. 网络推理：将 $\bar{s}_t^\tau$ 输入 VLA 骨干网络（如 Transformer），预测出当前的速度场，进而计算出期望去噪方向。
3. 噪声评估：辅助的可学习噪声网络根据当前状态输出标准差 $\sigma_\tau$。
4. 采样与记录：采样随机噪声 $\epsilon$，计算下一步隐变量 $\mathbf{A}_t^{\tau+\delta}$。同时，将该步的对数似然记录到缓存中：

$$
\log p_\theta\left(\mathbf{A}_t^{\tau+\delta}\mid \mathbf{A}_t^\tau,\mathbf{o}_t\right)
$$

步骤 3：动作执行与奖励获取（外部循环终点）

1. 当内部去噪循环到达 $\tau=1$ 时，得到无噪声的确定性物理动作 $\mathbf{a}_t=\mathbf{A}_t^1$。
2. 将动作 $a_t$ 下发给环境执行。
3. 观测状态更新，并返回单步奖励，例如在抓取任务成功时返回稀疏的 $+1$ 奖励。

步骤 4：强化学习策略更新（RL 反向传播）

1. 收集到一个或多个 Episode 的完整轨迹后，计算累计优势函数（Advantage）。
2. 提取出步骤 2 中精确计算的联合对数似然总和 $\sum_{\tau}\log p_\theta$。
3. 应用 PPO 算法的截断代理目标函数（Surrogate Objective）计算策略梯度。
4. 更新 VLA 模型网络参数 $\theta$ 以及噪声网络参数，使得在相似观测下，能获取高环境奖励的去噪轨迹生成概率最大化。

## 四、该方法如何解决流匹配RL“脱离最优传输后步数激增”的问题

在传统的流匹配中，轨迹变弯曲会导致 ODE 求解器（如欧拉法、RK4）在积分时产生巨大的截断误差，从而必须将步长 $\Delta t$ 切得极小（即 NFE 激增）。Flow-Noise 从根本上绕开了 ODE 求解器。它将整个生成过程重构成了一个离散时间的 MDP。环境的转移函数被强行定义为：

$$
\mathbf{A}_t^{\tau+\delta}=\bar{a}_t^\tau
$$

这意味着，模型不是在预测一个连续的“切线方向”，而是直接跳跃到下一个离散状态。RL 算法（如 PPO）优化的是这一个个离散跳跃点的累积奖励。既然根本不用连续微分方程来积分，自然也就无所谓“曲率导致截断误差”的问题，模型依然可以在较少的步数内（例如 10 步 MDP）稳定生成高奖励结果。

这让Flow-Noise看起来很像扩散模型，但同时又具备了流匹配模型自身去噪轨迹接近直线、速度快的优点。

## 参考文献

- Chen, K., Liu, Z., Zhang, T., Guo, Z., Xu, S., Lin, H., Zang, H., Li, X., Zhang, Q., Yu, Z., Fan, G., Huang, T., Wang, Y., & Yu, C. (2025). [π_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models](https://arxiv.org/abs/2510.25889). arXiv:2510.25889.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
