---
title: "10.1 Multi-Armed Bandits and the Upper Confidence Bound (UCB) Algorithm"
chapter_title: "Fundamentals of Reinforcement Learning"
section_id: "10-01"
language: en
source_language: zh
source_docx: "第2部分 强化学习/10.强化学习的基本知识/10.1 多臂老虎机与上置信界（UCB）算法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 10.1 Multi-Armed Bandits and the Upper Confidence Bound (UCB) Algorithm

## I. Multi-Armed Bandits (MAB)

In the multi-armed bandit problem (see Figure 2-1), a slot machine has K levers, each associated with a reward probability distribution R. Each time we pull a lever, we receive a reward r from its distribution. Without knowing the levers' reward distributions, we start from scratch and seek the highest possible cumulative reward after T pulls. Because the distributions are unknown, we must balance “investigating each lever's chance of winning” (exploration) against “choosing the most rewarding lever based on experience” (exploitation). The multi-armed bandit problem asks which action policy maximizes cumulative reward.

Formal description:

The multi-armed bandit problem can be represented by a tuple $(\mathcal{A}, \mathcal{R})$, where:

- $\mathcal{A}$ is the action set, with each action representing pulling a lever. If there are $K$ levers, the action space is $\{a_1,\ldots,a_K\}$; $a_t \in \mathcal{A}$ denotes an arbitrary action.
- $\mathcal{R}$ is the reward probability distribution. Each lever-pulling action $a$ corresponds to a distribution $R(r\mid a)$, and different levers usually have different reward distributions.

Assuming only one lever can be pulled at each time step, the objective is to maximize cumulative reward over $T$ steps:

$$
\max \sum_{t=1}^{T} r_t,\qquad r_t \sim R(\cdot\mid a_t).
$$

Here $a_t$ is the action of pulling a lever at time $t$, and $r_t$ is the reward obtained from $a_t$.

For each action $a$, define its expected reward as

$$
Q(a)=\mathbb{E}_{r\sim R(\cdot\mid a)}[r].
$$

At least one lever therefore has expected reward no smaller than any other lever. Denote this optimal expected reward by

$$
Q^{*}=\max_{a\in\mathcal{A}} Q(a).
$$

To see the gap between a lever's expected reward and that of the optimal lever more intuitively, we introduce “regret.” Regret is the difference between the expected rewards of the current and optimal actions:

$$
R(a)=Q^{*}-Q(a).
$$

Cumulative regret is the total regret after $T$ lever pulls. For a complete sequence of $T$ decisions $\{a_1,a_2,\ldots,a_T\}$, it is

$$
\mathcal{R}_T=\sum_{t=1}^{T} R(a_t).
$$

The MAB objective of maximizing cumulative reward is equivalent to minimizing cumulative regret.

To determine which lever yields a higher reward, we need to estimate its expected reward. A single pull produces a random reward, so we must pull the same lever repeatedly and calculate the expectation of the resulting rewards. An incremental estimation procedure is:

1. For every $a\in\mathcal{A}$, initialize counter $N(a)=0$ and expected-reward estimate $\hat{Q}(a)=0$.
2. For $t=1,\ldots,T$:
   - Choose a lever, denoting the action $a_t$.
   - Receive reward $r_t$.
   - Update the counter: $N(a_t)=N(a_t)+1$.
   - Update the expected-reward estimate: $\hat{Q}(a_t)\leftarrow \hat{Q}(a_t)+\frac{1}{N(a_t)}[r_t-\hat{Q}(a_t)]$.

Note: summing all numbers and dividing by their count has the disadvantage of $O(n)$ time and space complexity per update. Incremental updates require only $O(1)$ time and space.

The simplest policy is to keep taking the first action, but this depends heavily on luck. With excellent luck, that lever may have the maximum expected reward and be optimal; with terrible luck, it may yield the minimum expected reward.

A classic issue in MAB is balancing exploration and exploitation. Exploration means trying more possible levers. A lever need not yield the highest reward, but this approach reveals how all levers perform. For example, in a 10-armed bandit, we must try every lever to learn which may have the greatest reward. Exploitation means pulling the lever with the highest known expected reward. Since this knowledge comes only from limited interactions, the currently best lever need not be globally optimal. For example, after trying only three of 10 levers, we might keep pulling the best of those three, even though the best lever may be among the remaining seven. Even if we try all 10 levers 20 times each and find that lever 5 has the highest empirical expected reward, there remains a small probability that lever 6's true expected reward is higher.

Policies for MAB must therefore balance the number of exploratory and exploitative actions to maximize cumulative reward. A common approach is to explore more initially and exploit once each lever has a reasonably accurate estimate. Classic algorithms include epsilon-greedy, upper confidence bound, and Thompson sampling.

## II. Epsilon-Greedy Algorithm

A fully greedy algorithm always takes the action (pulls the lever) with the highest expected reward, performing pure exploitation without exploration. We therefore usually modify it; a classic modification is the $\epsilon$-greedy algorithm.

The $\epsilon$-greedy algorithm adds noise to the fully greedy algorithm: with probability $1-\epsilon$, it selects the lever with the highest expected reward according to experience (exploitation), and with probability $\epsilon$, it selects a random lever (exploration):

$$
a_t=
\begin{cases}
\arg\max\limits_{a\in\mathcal{A}} \hat{Q}(a), & \text{with probability } 1-\epsilon,\\
\text{choose randomly from }\mathcal{A}, & \text{with probability } \epsilon.
\end{cases}
$$

### (3) Upper Confidence Bound (UCB) Algorithm

Imagine a two-armed bandit: the first lever has been pulled only once, while the second has been pulled many times and its reward distribution is reasonably understood. What would you do? You might try the first lever again to become more certain about its reward distribution. This reasoning is based mainly on uncertainty: because the first lever has been tried only once, its uncertainty is high. Greater uncertainty makes a lever more valuable to explore, since exploration may reveal a high expected reward. Introduce an uncertainty measure U(a), which decreases as an action is tried more often. An uncertainty-based policy can jointly consider current expected-reward estimates and uncertainty; the key question is how to estimate uncertainty.

The upper confidence bound (UCB) algorithm is a classic uncertainty-based policy algorithm using a famous mathematical principle, Hoeffding's inequality. Let $X_1,\ldots,X_n$ be $n$ independent, identically distributed random variables in $[0,1]$, with empirical expectation

$$
\bar{x}_n=\frac{1}{n}\sum_{j=1}^{n} X_j,
$$

Then

$$
\Pr\{\mathbb{E}[X]\ge \bar{x}_n+u\}\le e^{-2nu^{2}}.
$$

Apply Hoeffding's inequality to MAB. Substitute $\hat{Q}_t(a)$ for $\bar{x}_n$; parameter $u=\hat{U}_t(a)$ represents uncertainty. Given probability

$$
p=e^{-2N_t(a)\hat{U}_t(a)^{2}},
$$

The inequality implies that $Q_t(a)<\hat{Q}_t(a)+\hat{U}_t(a)$ holds with probability at least $1-p$. For small $p$, $Q_t(a)<\hat{Q}_t(a)+\hat{U}_t(a)$ holds with high probability, so $\hat{Q}_t(a)+\hat{U}_t(a)$ is an upper bound on expected reward.

UCB then selects the action with the largest upper bound:

$$
a_t=\arg\max_{a\in\mathcal{A}}[\hat{Q}_t(a)+\hat{U}_t(a)].
$$

The “misjudgment” (the “rare event” we seek to avoid when setting $p$) is precisely that the lever's true expected reward exceeds “our observed average plus the uncertainty bonus.” Mathematically, we regard the following event as a “misjudgment”:

$$
Q(a)>\hat{Q}_t(a)+\hat{U}_t(a).
$$

To reduce probability $p$ of “mistaking a good lever for a bad one and missing a good solution,” we loosen $u$ and raise the upper confidence bound, encouraging more exploration.

Once p is set, uncertainty u is:

$$
u=\sqrt{\frac{-\ln p}{2n}}.
$$

We generally set $p=1/t$. When $t$ is small, the algorithm is encouraged to find an approximately optimal solution quickly with limited exploration. When $t$ is large, if an arm has not been selected for a long time ($n$ is unchanged) while total time $t$ increases, its exploration bonus grows. This forcibly raises its score, prompting the algorithm to “try it again” to meet increasingly strict confidence requirements. Otherwise, every selection reduces $u$ slightly; although “unpopular arms” retain larger $u$, all arms' $u$ gradually approach $0$ as time passes and $n$ increases, becoming negligible relative to $Q$ and eventually ineffective, potentially causing convergence to a suboptimal solution.

This can be written in the classic UCB form:

$$
a_t=\arg\max_{a\in\mathcal{A}}[\hat{Q}_t(a)+\sqrt{\frac{2\ln t}{N_t(a)}}].
$$

Technical aside: more rigorous UCB1 derivations often choose $p=t^{-4}$, yielding the exploration term $\sqrt{\frac{2\ln t}{N_t(a)}}$. If $p=1/t$, then $\sum_t 1/t$ diverges, whereas the series for $p=t^{-4}$ converges, making regret bounds easier to prove. Regardless of whether $1/t$ or $t^{-4}$ is used, the key is maintaining long-term exploration through $\ln t$.

## IV. Thompson Sampling

1. Core idea

If UCB is a rigorous mathematician (computing strict bounds and refusing to overlook any possibility), Thompson sampling is a wise gambler (simulating the future using probability distributions and confronting uncertainty with randomness). UCB is frequentist: it computes a fixed numerical value from observed data (mean plus error term). Thompson sampling is Bayesian. Its core logic is: rather than directly estimate a lever's winning probability, maintain a “probability distribution over its winning probability.” At every step, randomly draw a number from each lever's distribution, pretend it is that lever's true ability, and choose the lever with the largest sampled value.

For an unfamiliar lever: I do not know its winning probability, so I consider any value from 0 to 1 possible (a broad, flat distribution).

For a familiar lever: I know its winning probability is about 80%, so I believe its distribution concentrates mainly between 0.75 and 0.85 (a narrow, sharp distribution).

2. Mathematical model: the beta distribution

For bandits with binary rewards (win/lose, $1/0$), Thompson sampling usually uses a beta distribution as a conjugate prior.

Do not be intimidated by the name; simply remember its two parameters, $\alpha$ and $\beta$.

$$
f(x;\alpha,\beta)=\mathrm{Beta}(\alpha,\beta)
$$

Their meanings in a bandit setting are extremely simple:

- $\alpha$ (alpha): the number of times the lever “wins (reward 1)” $+1$.
- $\beta$ (beta): the number of times the lever “loses (reward 0)” $+1$.

Why use this?

The shape of the beta distribution determines our view of the lever:

1. Initial state ($\alpha=1,\beta=1$): a uniform distribution. Every winning probability in $[0,1]$ seems equally likely. This represents “ignorance.”
2. Many wins, few losses ($\alpha=100,\beta=10$): the curve peaks strongly near $0.9$, meaning “I am very certain this is a good lever.”
3. Very few trials ($\alpha=2,\beta=1$): the curve is relatively flat, meaning “It could be very good or very bad; uncertainty is high.”

3. Algorithm

Thompson sampling is very concise and does not require calculating complicated square roots and logarithms as UCB does:

1. Initialize: assign counters $\alpha_k=1,\beta_k=1$ to every lever $k$.
2. At each step $t$:
   - Sample: for every lever $k$, draw a random value $\theta_k\sim \mathrm{Beta}(\alpha_k,\beta_k)$ from its distribution $\mathrm{Beta}(\alpha_k,\beta_k)$.

   - Select: choose the lever with the largest $\theta_k$ (denote the action $A_t$).
   - Observe: pull $A_t$ and obtain reward $r$ ($0$ or $1$).
   - Update: update the parameters (Bayesian inference):
     - If it wins ($r=1$): $\alpha_{A_t}\leftarrow \alpha_{A_t}+1$
     - If it loses ($r=0$): $\beta_{A_t}\leftarrow \beta_{A_t}+1$

## References

- [Hands-on Reinforcement Learning (translated title; in Chinese)](https://hrl.boyuai.com/).
- Thompson, W. R. (1933). [On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples](https://doi.org/10.1093/biomet/25.3-4.285). Biometrika, 25(3/4), 285–294.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352). Machine Learning.
