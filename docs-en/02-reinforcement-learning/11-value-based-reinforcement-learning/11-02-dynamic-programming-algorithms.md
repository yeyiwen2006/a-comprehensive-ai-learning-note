---
title: "11.2 Dynamic Programming Algorithms"
chapter_title: "Value-Based Reinforcement Learning"
section_id: "11-02"
language: en
source_language: zh
source_docx: "第2部分 强化学习/11.基于价值的强化学习/11.2 动态规划算法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 11.2 Dynamic Programming Algorithms

Dynamic programming is an important part of algorithm design, efficiently solving classic problems such as the knapsack problem and shortest-path planning. Its basic idea is to decompose a problem into subproblems, solve them first, and obtain the original solution from their solutions. It stores solved subproblem results for direct reuse when needed, avoiding repeated computation. This chapter introduces how to use dynamic programming to find optimal policies in Markov decision processes.

Dynamic-programming-based reinforcement learning solves for a policy when environment transitions and rewards are known. There are two main approaches: policy iteration and value iteration. Policy iteration consists of policy evaluation and policy improvement. Specifically, policy evaluation uses the Bellman expectation equation to obtain a policy's state-value function through dynamic programming; value iteration directly applies dynamic programming to the Bellman optimality equation to obtain the final optimal state values.

Unlike the Monte Carlo methods introduced earlier and temporal-difference algorithms introduced later, these two dynamic-programming-based RL algorithms require prior knowledge of the transition and reward functions, meaning the entire MDP must be known. In such a white-box environment, state values can be solved directly by dynamic programming without extensive agent–environment interaction. Real-world white-box environments are rare, however, limiting dynamic programming's practical applicability. Policy and value iteration are also usually limited to finite MDPs with discrete, finite state and action spaces.

### 1. Policy Iteration (the Basis of Sarsa)

(1) Policy evaluation

Policy evaluation calculates a policy's state-value function. Recall the Bellman expectation equation:

$$
V^{\pi}(s)=\sum_{a\in A}\pi(a|s)(r(s,a)+\gamma\sum_{s'\in S}p(s'|s,a)V^{\pi}(s'))
$$

Here $\pi(a|s)$ is the probability of action $a$ in state $s$ under $\pi$. Given rewards and transitions, the current state's value can be computed from the next state's value. Following dynamic programming, calculating possible next-state values is a subproblem, while calculating the current value is the current problem. Once subproblem solutions are known, the current problem can be solved. More generally, across all states, the previous iteration's value function is used to calculate the current one:

$$
V^{k+1}(s)=\sum_{a\in A}\pi(a|s)(r(s,a)+\gamma\sum_{s'\in S}P(s'|s,a)V^{k}(s'))
$$

Choose any initial $V^{0}$. The Bellman expectation equation shows that $V^{k}=V^{\pi}$ is a fixed point of this update. In fact, as $k\to\infty$, the sequence $\{V^{k}\}$ can be proved to converge to $V^{\pi}$, allowing the policy's state-value function to be calculated. Repeated Bellman expectation updates make policy evaluation computationally expensive. In practice, evaluation can stop early once $\max_{s\in S}|V^{k+1}(s)-V^{k}(s)|$ becomes very small. This improves efficiency while producing values very close to their true values.

(2) Policy improvement

After policy evaluation gives the current state-value function, it can improve the policy. Suppose the value $V^{\pi}$ of policy $\pi$ is known, giving the expected return from every state $s$ under $\pi$. How can the policy change to obtain a higher expected return at $s$? If the agent takes action $a$ at $s$ and thereafter follows $\pi$, the expected return is action value $Q^{\pi}(s,a)$. If $Q^{\pi}(s,a)>V^{\pi}(s)$, taking $a$ at $s$ yields higher expected return than the original policy $\pi(a|s)$. This concerns one state. Now suppose a deterministic policy $\pi'$ satisfies, for every state $s$,

$$
Q^{\pi}(s,\pi'(s))\ge V^{\pi}(s)
$$

Then for every state $s$,

$$
V^{\pi'}(s)\ge V^{\pi}(s)
$$

This is the policy improvement theorem. We can therefore greedily choose the highest-action-value action in every state:

$$
\pi'(s)=\arg\max_a Q^{\pi}(s,a)=\arg\max_a\bigl(r(s,a)+\gamma\sum_{s'}P(s'|s,a)V^{\pi}(s')\bigr)
$$

The constructed greedy policy $\pi'$ meets the theorem's conditions, so policy $\pi'$ is better than or at least as good as $\pi$. Selecting actions greedily to obtain a new policy is called policy improvement. If the improved $\pi'$ equals the previous $\pi$, policy iteration has converged, and $\pi$ and $\pi'$ are optimal.

Proof of the policy improvement theorem: the following derivation shows that the new $\pi'$ obtained from the improvement formula has value no lower than $\pi$ in every state.

$$
\begin{aligned}
V^{\pi}(s) &\le Q^{\pi}(s,\pi'(s))\\
&= \mathbb{E}_{\pi'}[R_t+\gamma V^{\pi}(S_{t+1})|S_t=s]\\
&\le \mathbb{E}_{\pi'}[R_t+\gamma Q^{\pi}(S_{t+1},\pi'(S_{t+1}))|S_t=s]\\
&= \mathbb{E}_{\pi'}[R_t+\gamma R_{t+1}+\gamma^{2}V^{\pi}(S_{t+2})|S_t=s]\\
&\le \mathbb{E}_{\pi'}[R_t+\gamma R_{t+1}+\gamma^{2}R_{t+2}+\gamma^{3}V^{\pi}(S_{t+3})|S_t=s]\\
&\vdots\\
&\le \mathbb{E}_{\pi'}[R_t+\gamma R_{t+1}+\gamma^{2}R_{t+2}+\gamma^{3}R_{t+3}+\cdots|S_t=s]\\
&= V^{\pi'}(s)
\end{aligned}
$$

Every time step uses the local action-value advantage $V^{\pi}(S_{t+1})\le Q^{\pi}(S_{t+1},\pi'(S_{t+1}))$. Accumulating to infinitely many steps or a terminal state gives the inequality for improvement of the entire policy value.

(3) Policy iteration

Overall, policy iteration evaluates the current policy to obtain its state-value function, improves the policy using those values, and then repeatedly evaluates and improves the new policy until convergence to an optimal policy (see Section 4.7 for the convergence proof):

$$
\pi^{0} \xrightarrow{\text{policy evaluation}} V^{\pi^{0}} \xrightarrow{\text{policy improvement}} \pi^{1} \xrightarrow{\text{policy evaluation}} V^{\pi^{1}} \xrightarrow{\text{policy improvement}} \pi^{2} \xrightarrow{\text{policy evaluation}} \cdots \xrightarrow{\text{policy improvement}} \pi^{*}
$$

Combining evaluation and improvement gives the policy iteration algorithm:

- Randomly initialize policy $\pi(s)$ and value function $V(s)$.
- While $\Delta>\theta$, perform the policy-evaluation loop:
  - $\Delta\leftarrow 0$
  - For every state $s\in S$:
    - $v\leftarrow V(s)$
    - $V(s)\leftarrow r(s,\pi(s))+\gamma\sum_{s'}P(s'|s,\pi(s))V(s')$
    - $\Delta\leftarrow \max(\Delta,|v-V(s)|)$
- After evaluation, set $\pi_{old}\leftarrow\pi$.
- For every state $s\in S$:
  - $\pi(s)\leftarrow \arg\max_a r(s,a)+\gamma\sum_{s'}P(s'|s,a)V(s')$
- If $\pi_{old}=\pi$, stop and return $V$ and $\pi$; otherwise return to policy evaluation.

Relationship to Sarsa:

During policy evaluation, suppose the value of taking action $a$ in state $s$ under the current policy is:

$$
Q^{\pi}(s,a)=\sum_{s'}P(s'|s,a)[R(s,a)+\gamma Q^{\pi}(s',\pi(s'))]
$$

Note: this uses $Q^{\pi}(s',\pi(s'))$ or $\sum_{a'}\pi(a'|s')Q(s',a')$, not $\max_{a'}Q(s',a')$. It faithfully evaluates the current policy's performance.

Here $\pi(s')$ denotes the $Q$ obtained assuming the next step still follows the same, pre-update policy.

Greedy updates occur afterward, during policy improvement.

In Sarsa, the distribution $P(s'|s,a)$ is unknown, so expectations cannot be computed directly; only “sampling plus a moving average” is available:

Sarsa stands for State-Action-Reward-State-Action. Its update is:

$$
Q(s,a)\leftarrow Q(s,a)+\alpha[\underbrace{R+\gamma Q(s',a')}_{\text{TD Target}}-Q(s,a)]
$$

- The connection:
  - Sarsa's target $R+\gamma Q(s',a')$ uses the next action $a'$ actually taken by the agent.
  - It therefore estimates the value $Q^{\pi}$ of the current policy $\pi$ rather than optimal value $Q^{*}$, corresponding to policy iteration's “policy evaluation” step.
  - During training, $\epsilon$ is usually gradually reduced (making the policy increasingly greedy). This continually improves the policy, corresponding to policy iteration's “policy improvement” step.

### 2. Value Iteration (the Basis of Q-Learning)

More precisely, value iteration is a dynamic-programming process using the Bellman optimality equation:

$$
V^{*}(s)=\max_{a\in A}\bigl(r(s,a)+\gamma\sum_{s'\in S}P(s'|s,a)V^{*}(s')\bigr)
$$

Written as an iterative update:

$$
V^{k+1}(s)=\max_{a\in A}\bigl(r(s,a)+\gamma\sum_{s'\in S}P(s'|s,a)V^{k}(s')\bigr)
$$

Value iteration follows this update. Once $V^{k+1}$ equals $V^{k}$, it is a fixed point of the Bellman optimality equation, corresponding to optimal state values $V^{*}$. Recover the optimal policy using $\pi(s)=\arg\max_a(r(s,a)+\gamma\sum_{s'}p(s'|s,a)V^{k+1}(s'))$.

The value iteration algorithm is:

- Randomly initialize $V(s)$.
- While $\Delta>\theta$:
  - $\Delta\leftarrow 0$
  - For every state $s\in S$:
    - $v\leftarrow V(s)$
    - $V(s)\leftarrow \max_a r(s,a)+\gamma\sum_{s'}P(s'|s,a)V(s')$
    - $\Delta\leftarrow \max(\Delta,|v-V(s)|)$
- After the loop, return a deterministic policy $\pi(s)=\arg\max_a(r(s,a)+\gamma\sum_{s'}P(s'|s,a)V(s'))$.

Relationship to Q-Learning:

With a known environment model (transition probabilities $P$ and rewards $R$), value iteration directly computes optimal action values $Q^{*}(s,a)$ iteratively. Its expectation-based update is:

$$
Q_{k+1}(s,a)=\sum_{s'}P(s'|s,a)[R(s,a)+\gamma\max_{a'}Q_k(s',a')]
$$

Q-Learning does not know this distribution and can only use “sampling plus a moving average”:

$$
Q(s,a)\leftarrow Q(s,a)+\alpha[\underbrace{R+\gamma\max_{a'}Q(s',a')}_{\text{TD Target}}-Q(s,a)]
$$

## References

- [Hands-on Reinforcement Learning (translated title; in Chinese)](https://hrl.boyuai.com/).
