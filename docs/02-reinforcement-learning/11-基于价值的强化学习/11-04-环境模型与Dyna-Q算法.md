---
title: "11.4 环境模型与Dyna-Q算法"
source_docx: "第2部分 强化学习/11.基于价值的强化学习/11.4 环境模型与Dyna-Q算法.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 11.4 环境模型与Dyna-Q算法


## 一、无环境模型的Q-Learning存在的问题

1.反向传播的滞后性

在标准的 Q-Learning（Model-Free）中，价值函数的更新公式是：

$$
Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma\max_{a^{\prime}}Q(S^{\prime},a^{\prime})-Q(S,A)]
$$

请注意，这个更新只发生在一步之间。

举个例子（迷宫寻宝）：假设路径是起点 $S_0\to S_1\to S_2\to\cdots\to S_{99}\to$ 宝藏（+100）。

1. 第一次尝试（Episode 1）：智能体瞎逛终于走到了宝藏。
   - 只有最后一步 $S_{99}\to$ 宝藏获得了奖励。
   - 结果：只有 $Q(S_{99},动作)$ 被更新了，知道了这里有好东西。
   - 问题：$S_{98}$、$S_{50}$ 甚至起点 $S_0$ 的 $Q$ 值依然是 0。智能体在 $S_{99}$ 之前完全不知道自己走对了。
2. 第二次尝试（Episode 2）：智能体再次从起点出发。
   - 它必须再次随机走到 $S_{99}$。
   - 只有当它从 $S_{98}$ 走到 $S_{99}$ 时，看到 $S_{99}$ 的 $Q$ 值很高，$Q(S_{98},动作)$ 才会更新。
   - 问题：价值从终点传导到起点，需要走 100 个完整的 Episode。这简直慢得令人发指。

反向传播的滞后性导致更新严重偏慢，而且还会引导致其对环境变化的反应迟钝：如果环境发生了微小的变化（例如之前的路不通了），智能体往往难以快速调整策略，会在这个新环境中撞得头破血流很多次。

2.样本效率低下

为了获得准确的Q值，智能体需要大量次数与环境交互并获取样本。但现实中获取大量样本往往是昂贵的：在机器人控制中，每走一步都要耗电、磨损机械臂，甚至可能摔坏；在金融交易中，每一次试错都是真金白银。相比之下，一次计算成本极低。

## 二、Dyna-Q算法：一种基于环境模型（基于模型）的强化学习算法

1.步骤

Dyna-Q 的一次完整迭代包含以下四个关键步骤。

**步骤 1：与环境交互（Acting）**

智能体处于状态 $S$，根据当前的 $Q$ 表（通常使用 $\epsilon$-greedy 策略）选择动作 $A$，执行动作后获得奖励 $R$ 和新状态 $S'$。

$$
(S,A)\xrightarrow{\text{Environment}}(R,S')
$$

**步骤 2：直接强化学习（Direct Reinforcement Learning）**

利用刚才获得的真实经验 $(S,A,R,S')$，使用 Q-Learning 的更新公式更新 $Q$ 值：

$$
Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma\max_{a^{\prime}}Q(S^{\prime},a^{\prime})-Q(S,A)]
$$

这一步保证了算法具有 Model-Free 算法的无偏性。

**步骤 3：模型学习（Model Learning）**

利用真实经验更新环境模型。对于确定性环境，我们只需简单地记录结果：

$$
\mathrm{Model}(S,A)\leftarrow(R,S^{\prime})
$$

这意味着：如果将来在“想象”中遇到状态 $S$ 并尝试动作 $A$，模型将告诉智能体结果是 $R$ 和 $S'$。

环境模型（世界模型）是一个函数，输入 $s_t$ 和 $a_t$，该函数会输出 $r_t$ 和 $s_{t+1}$。这个函数可以是一个神经网络。

**步骤 4：规划（Planning）**

这是 Dyna-Q 区别于 Q-Learning 的核心。智能体利用模型进行 $N$ 次模拟更新：

Repeat $N$ times:

1. 随机选择状态 $\tilde{S}$：从之前观测过的状态集中随机选取一个。
2. 随机选择动作 $\tilde{A}$：从在状态 $\tilde{S}$ 下执行过的动作集中随机选取一个。
3. 模型预测：将 $(\tilde{S},\tilde{A})$ 输入模型，得到预测的奖励 $\tilde{R}$ 和下一状态 $\tilde{S}'$：

$$
(\tilde{R},\tilde{S}^{\prime})\leftarrow \mathrm{Model}(\tilde{S},\tilde{A})
$$

4. 模拟更新（Simulated Update）：再次利用 Q-Learning 公式更新 $Q$ 值，但这使用的是模拟数据：

$$
Q(\tilde{S},\tilde{A})\leftarrow Q(\tilde{S},\tilde{A})+\alpha[\tilde{R}+\gamma\max_{a^{\prime}}Q(\tilde{S}^{\prime},a^{\prime})-Q(\tilde{S},\tilde{A})]
$$

在高级版本的环境模型中，我们未必要让它选择走过的 $(s_t,a_t)$，而是可以更随机地选择。由于环境模型学习了环境的规律，它往往具有泛化性能，可以输出一个预测值。

（2）补充解释

假设一个迷宫环境，智能体找到目标获得奖励。

Q-Learning：由于智能体运用马尔可夫决策模型，只有当智能体再次走到目标附近时，奖励信息才会通过Q值反向传播一步。

Dyna-Q：智能体哪怕还在起点附近，通过N次规划，它可以在“脑海”中复盘之前的记忆。如果它经历过从起点到终点，Planning步骤会利用模型产生的模拟转换，将终点的高价值迅速向起点方向传播。

这就像我们在做题（真实环境），做完一道题后，我们会回忆（模型预测）之前做过的类似题目（模拟经验），从而加深理解，不需要把所有题目再做一遍。

（3）Dyna-Q的问题：模型偏差（Model Bias）

Dyna-Q强依赖于模型的准确性。如果Model(s, a)预测的结果与真实环境不符（例如环境发生了变化，但模型还没更新），Planning 步骤实际上是在传播错误的知识，这会导致智能体制定出在模型中看似最优、但在现实中行不通的策略。

为了解决环境变化导致模型失效的问题，后继研究提出了 Dyna-Q+，给模型中那些“长时间未被访问”的状态-动作对添加额外的探索奖励（Exploration Bonus）：

公式调整：在规划阶段，模拟奖励 $\tilde{R}$ 被修正为 $\tilde{R}+\kappa\sqrt{\tau}$，其中 $\tau$ 是该状态-动作对上次被访问至今的时间步数，$\kappa$ 是系数。

这鼓励智能体去验证那些很久没去过的地方，从而及时更新模型，修正偏差。

## 参考文献

- Sutton, R. S. (1990). [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](http://incompleteideas.net/papers/sutton-90.pdf). ICML 1990.
