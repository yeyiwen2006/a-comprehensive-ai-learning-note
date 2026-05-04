---
title: "13.1 Actor-Critic算法"
source_docx: "第2部分 强化学习/13.综合价值与策略的算法/13.1 Actor-Critic算法.docx"
status: "auto-converted"
ocr: "disabled; image content awaits manual reconstruction"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 13.1 Actor-Critic算法


## 一、Actor-Critic算法的基本原理

1.提出背景

在策略梯度算法中，我们通过策略梯度获得了参数θ的调整方向，再通过动作价值的蒙特卡洛估计G_t来获得调整的正负和幅度。但是蒙特卡洛估计存在需要样本量高、必须走完全程才能更新、方差大等问题。考虑到G_t本身其实是对动作价值优劣的估计，我们可以结合前面基于价值的强化学习方法，引入动作价值函数。

2.数学表示

为了让目标函数（Q值的在状态-动作对概率分布下的期望）最大，策略梯度公式应为：

回顾我们之前得出的结论，最理想的策略梯度公式是：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a\mid s)\cdot Q^\pi(s,a)\right]
$$

为了降低方差，我们通常会引入一个基线（Baseline）$V^\pi(s)$，将 $Q$ 转化为优势函数（Advantage Function）：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

这里Q(s,a)表示s状态下采取a动作的价值，V(s)表示s状态的价值，A(s,a)就是动作a相对于平均水平的优势。

根据贝尔曼方程，在一次采样中，如果接下来走到s’，则下一步更新中认为：

根据贝尔曼方程（Bellman Equation）：

$$
Q(s,a)\approx r+\gamma V(s')
$$

将这个代入优势函数公式：

$$
A(s,a)\approx \underbrace{r+\gamma V_\phi(s')-V_\phi(s)}_{\text{TD Error }\delta}
$$

从而我们并不需要构建Q网络，只需要一个V网络即可。

3.网络更新

和Sarsa不同：没有“Q表”

Actor-Critic算法通常由策略网络和价值网络两部分构成，将策略函数的优化与价值函数的优化分开，从而结合了两种学习的优点。

（1）策略网络（Actor）的更新

策略网络由上面的策略梯度表达式更新。我们把策略梯度中的G_t换成了这种策略下的A(s,a)值。由于我们不需要G_t，而是可以直接利用价值网络存储的A(s,a)，故每走一步，就可以进行一次更新。当A(s,a)＞0时，会倾向于让a的概率增大，反之倾向于减小。

（2）价值网络（Critic）的更新

当从状态s到s’，获得奖励r，V(s)的更新目标Target=r+gamma*V(s’)。

4.与其他算法（Sarsa、DQN）的对比

（1）与Sarsa的对比

Sarsa的Q既负责打分，也负责决策（epsilon-greedy策略），而Critic的Q只负责打分，策略权在Actor手里；Sarsa有显式Q表，Critic通过深度学习拟合。

相同点（The Core DNA）：

* 数学根据一致：都基于贝尔曼期望方程（Bellman Expectation Equation），即 $\mathrm{Value}\approx r+\gamma\cdot\mathrm{Next\_Value}$。这里的 $\mathrm{Next\_Value}$ 都是指“按照当前策略走下去的价值”，不取最大值。
* 策略属性一致：通常都是 On-policy（同策略）。它们更新的目标值，紧紧跟随当前智能体的实际行为模式；如果智能体变笨了，它们评估出的价值也会变低。

| 维度 | Critic（通常指 V-based） | Sarsa（Q-based） |
| --- | --- | --- |
| 所需数据 | $S,A,R,S'$ | $S,A,R,S',A'$ |
| 更新时机 | 看到状态 $S'$ 即可更新。它只需要预估 $S'$ 的平均前景。 | 必须等到选出 $A'$ 后。它必须知道下一步确切走了哪条路。 |
| 更新目标 | $Target=r+\gamma V(s')$，直接使用状态价值，隐含了对动作的期望。 | $Target=r+\gamma Q(s',a')$，使用下一步实际发生动作的价值。 |
| 随机性/方差 | 较低。$V(s')$ 是网络输出的平滑值，过滤掉了下一步选动作的随机噪声。 | 较高。受下一步动作 $a'$ 的具体选择影响，可能这步走得很好、下步走得烂，导致 Target 波动。 |
| 核心角色 | 辅助者（Coach）。它只负责打分，必须配合 Actor 使用，自己不产出动作。 | 独裁者（Doer）。既负责打分，又直接根据分值决定动作（$\epsilon$-greedy）。 |

（2）与DQN的对比

Critic和Sarsa运用的都是贝尔曼期望方程，而DQN运用的是贝尔曼最优方程。Critic和Sarsa根据当前策略对应的概率分布，估计价值的期望，数据必须实时由当前策略生成，具体体现：

Sarsa：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left[\underbrace{r+\gamma Q(s',a')}_{\text{Target}}-Q(s,a)\right]
$$

Critic：

$$
L=\left(\underbrace{r+\gamma V(s')}_{\text{Target}}-V(s)\right)^2
$$

虽然式子里看起来没有策略，但是这种更新方式意味着V(s)本质上是过去不同动作得到的Q(s,a)（即r+gamma*V(s’)）的移动平均。随着策略的变化，不同动作的概率分布也就会变化，因此在下一步，必须重新根据此刻的策略（动作概率分布）采样新的动作a，才能保证r+V(s’)的期望是对新的策略的价值的无偏估计，而不能用别的策略下采样的动作进行V的更新，因而它是On-policy的。

而DQN的更新与策略无关，只取最优价值。

| 特性 | Critic（in AC） | Sarsa | DQN |
| --- | --- | --- | --- |
| 核心方程 | 贝尔曼期望方程 | 贝尔曼期望方程 | 贝尔曼最优方程 |
| 目标值（Target） | $r+\gamma V(s')$ | $r+\gamma Q(s',a')$ | $r+\gamma\max Q(s',a')$ |
| 对未来的态度 | 诚实（基于当前概率） | 诚实（基于实际选择） | 乐观（假设未来最优） |
| 所需样本 | $(s,a,r,s')$ | $(s,a,r,s',a')$ | $(s,a,r,s')$ |
| 数据利用 | On-policy（通常不复用） | On-policy（通常不复用） | Off-policy（可复用经验池） |
| 主要用途 | 减小 PG 的方差（PPO/A2C） | 学习安全保守的策略 | 学习寻找最优解的策略 |

场景：一条路紧贴着悬崖。掉下去扣 100 分，走正路扣 1 分。

策略特点：现在的策略 $\pi$ 还有点笨，有 $10\%$ 的概率会手滑掉下去（$\epsilon$-greedy）。

DQN（Q-Learning）的视角：

* 它会说：“嘿，贴着悬崖走是最短路径！只要我不手滑（取 max），这就是最优解。”
* 结果：它学出的价值函数会鼓励你贴着悬崖走。但因为你实际上会手滑，你就会不断掉下去。
* 特点：盲目自信，评估的是“理论上的完美策略”。

SARSA（以及 AC 中的 Critic）的视角：

* 它会说：“嘿，虽然贴着悬崖走最近，但以你现在的笨拙程度（当前策略 $\pi$），你走那条路有很大几率掉下去摔死。所以那条路的价值（$Q$ 或 $V$）其实很低！”
* 结果：它给出的价值估计会逼迫 Actor 离悬崖远一点，走安全的路。
* 特点：实事求是，评估的是“你现在的真实水平”。

5.训练和前向推理时的组件

所有存在Actor和Critic的模型架构，包括后面介绍的TRPO、PPO、DDPG、SAC等，都遵循以下原则：

训练时：Actor和Critic同时工作。

前向推理时：舍弃Critic，只留Actor。

这就好比你学开车（训练）时，教练（Critic）坐在副驾不停地给你打分、纠正动作（更新策略）；但是当你拿到驾照自己上路（推理）时，教练就不在了，只有你（Actor）在开车，你只需要看路况（状态）并打方向盘（动作），动作也不需要再纠正（策略不再更新）。

## 二、A2C和A3C

由前面介绍的内容，这是一个On-Policy算法，而且无法像SAC那样建立Replay Buffer（原因见SAC部分），每次更新策略都必须重新采样。这不仅不适合并行计算，而且导致了严重的数据相关性问题，因为相近时刻的状态s也比较相近，随着训练进行，状态随时间的分布会改变，数据不满足独立同分布假设。故我们需要有不同Worker在同一时刻分别采样，统一更新梯度。具体有以下两种底层实现方法。

1.A3C（Google DeepMind,ICML 2016）

每一个Worker都是一个独立的进程，进程里都有一个环境副本、一个本地神经网络副本。Worker自己在本地网络跑前向传播，算出梯度，然后把梯度推送到全局网络，并把全局网络的最新参数拉回来覆盖本地网络。

2.A2C（OpenAI,2017）

使用multiprocessing也就是所谓的SubprocVecEnv（子进程向量化环境）。每一个Worker进程里只有一个环境副本，主进程持有唯一的一个神经网络（通常在GPU上）。16个Worker并行执行env.step()，返回各自状态。主进程收集16个s，拼接成一个Batch，用GPU一次性算出16个各自的动作，发回给Worker。Worker非常轻量，只负责模拟物理环境。

目前主流的Actor-Critic算法（包括PPO），其底层并行架构基本都沿用了A2C的“向量化环境”模式。

## 参考文献

- Mnih, V., Badia, A. P., Mirza, M., et al. (2016). [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). ICML 2016.
- OpenAI. (2017). [OpenAI Baselines: ACKTR & A2C](https://openai.com/index/openai-baselines-acktr-a2c/).
