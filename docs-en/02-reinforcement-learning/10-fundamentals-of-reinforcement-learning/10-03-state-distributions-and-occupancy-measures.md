---
title: "10.3 State Distributions and Occupancy Measures"
chapter_title: "Fundamentals of Reinforcement Learning"
section_id: "10-03"
language: en
source_language: zh
source_docx: "第2部分 强化学习/10.强化学习的基本知识/10.3 状态分布与占用度量.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 10.3 State Distributions and Occupancy Measures

## I. Definitions of State Distributions and Occupancy Measures

The state distribution is defined as the discounted probability that the agent occupies state $s$:

Define $\nu^\pi(s)$ as the discounted state-visitation distribution under policy $\pi$:

$$
\nu^\pi(s) = (1-\gamma)\sum_{t=0}^{\infty}\gamma^t P_t^\pi(s)
$$

$P_t^\pi(s)$: the probability that the agent occupies state $s$ at time $t$.

$\gamma$: the discount factor, $0 \le \gamma < 1$.

$1-\gamma$: the normalization factor.

We can then define the occupancy measure, the discounted probability of the agent taking action a in state s:

$$
\rho^\pi(s,a) = (1-\gamma)\sum_{t=0}^{\infty}\gamma^t P_t^\pi(s)\pi(a|s)
$$

This gives the relationship:

$$
\rho^\pi(s,a) = \nu^\pi(s)\pi(a|s)
$$

## II. Derivation

1. Why is a discount factor needed?

We usually seek to maximize cumulative discounted return:

$$
J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t R(s_t,a_t)\right]
$$

By linearity of expectation, move the sum outside the expectation:

$$
J(\pi) = \sum_{t=0}^{\infty}\gamma^t\mathbb{E}_\pi[R(s_t,a_t)]
$$

$$
J(\pi) = \sum_{t=0}^{\infty}\gamma^t\sum_{s,a}P_t^\pi(s)\pi(a|s)R(s,a)
$$

Now move the sum over $s$ and $a$ to the front:

$$
J(\pi) = \sum_{s,a}R(s,a)\left(\sum_{t=0}^{\infty}\gamma^t P_t^\pi(s)\pi(a|s)\right)
$$

The parenthesized term is the unnormalized occupancy measure. To turn it into a probability distribution (summing to 1), multiply and divide by $(1-\gamma)$:

$$
J(\pi) = \frac{1}{1-\gamma}\sum_{s,a}R(s,a)\left((1-\gamma)\sum_{t=0}^{\infty}\gamma^t P_t^\pi(s)\pi(a|s)\right)
$$

The parenthesized part is $\rho^\pi(s,a)$, so:

$$
J(\pi) = \frac{1}{1-\gamma}\sum_{s,a}\rho^\pi(s,a)R(s,a) = \frac{1}{1-\gamma}\mathbb{E}_{(s,a)\sim\rho^\pi}[R(s,a)]
$$

In reinforcement learning, the goal is often to find a good policy. A policy can be represented by a probability distribution, and the optimization objective is precisely to maximize the expectation of return $R$ under $\rho^\pi$. The problem thereby becomes finding such a distribution. A state-action pair's probability is its contribution weight to total value, consistent with our understanding of Markov decision processes.

2. Why is a normalization factor needed?

This is a crucial mathematical detail. We must prove that $\nu^\pi(s)$ is a valid probability distribution, summing to 1 over all states.

Sum over every state $s$:

$$
\sum_s \nu^\pi(s) = \sum_s\left[(1-\gamma)\sum_{t=0}^{\infty}\gamma^t P_t^\pi(s)\right]
$$

Interchange the sums (assuming convergence):

$$
= (1-\gamma)\sum_{t=0}^{\infty}\gamma^t\left(\sum_s P_t^\pi(s)\right)
$$

Whatever $t$ is, the agent must occupy some state, so $\sum_s P_t^\pi(s)=1$. The expression becomes:

$$
= (1-\gamma)\sum_{t=0}^{\infty}\gamma^t \cdot 1 = 1
$$

## III. Mathematical Meaning of Occupancy Measures: Sampling Time Geometrically

We usually understand “visitation probability” as the percentage of all time steps spent in a state over the long run (a simple frequency statistic).

With discount factor $\gamma$, however, $\rho^\pi(s,a)$ is clearly no longer a simple “time frequency,” because it emphasizes earlier times over later ones.

It is still called a “probability” because we have changed the probability distribution used to “sample time $t$.”

Traditional view (undiscounted, $\gamma=1$)

Every time $t$ is assumed equally likely to be sampled (a mathematically uniform distribution is actually impossible over infinite time, so a limit is usually taken). This leads to the familiar “stationary distribution,” simply the time-average proportion.

Discounted view (discounted, $\gamma<1$)

$\rho^\pi(s,a)$ actually defines a new random experiment: instead of sampling time uniformly, sample $t$ according to a geometric distribution.

Sampling rules:

- The probability of $t=0$ is $(1-\gamma)$.
- The probability of $t=1$ is $(1-\gamma)\gamma$.
- The probability of $t=2$ is $(1-\gamma)\gamma^2$.
- ...
- The probability of time $t$ is $P(T=t)=(1-\gamma)\gamma^t$, where $T$ is the sampled time.

Its meaning is now:

If the observation time is uncertain and an instant is drawn randomly according to this “sooner matters more” rule, what is the probability that the agent is at $(s,a)$ at that instant?

Another interpretation is:

For greater intuition, introduce random termination (also the classic physical interpretation of $\gamma$):

Suppose the game does not continue indefinitely. Instead, at the end of each step, God rolls a die:

- With probability $\gamma$, the game continues.
- With probability $1-\gamma$, the game suddenly ends.

Then $\rho^\pi(s,a)$ measures:

The probability that the agent is in state $s$ and taking action $a$ in the game-over snapshot, at the instant the game “suddenly ends”.

This interpretation matches the formula perfectly:

- The probability that the game ends exactly at $t$ is $(1-\gamma)\gamma^t$.
- The probability that the agent is at $(s,a)$ at $t$ is $P_t^\pi(s,a)$.
- Summing over all possible termination times gives the total probability of observing $(s,a)$ at termination.

Because the game must end at some point, these probabilities sum to 1 over all $(s,a)$. It is therefore a proper probability distribution.

## IV. Recursive Representation of the State Distribution

When calculating $\nu^\pi(s')$, we sum the weighted occurrences of the agent at $s'$ across all time steps from $t=0$ to $t=\infty$.

Every occurrence has one of two possible timings:

1. It occurs at $t=0$: the “initial state.”
2. It occurs at $t>0$: a “transition from the previous time step.”

Total probability at $s'$ = weight of starting there + weight of transitions from elsewhere.

$$
\nu^\pi(s') = (1-\gamma)\nu_0(s') + \gamma\int P(s'|s,a)\pi(a|s)\nu^\pi(s)\,ds\,da
$$

The first term $(1-\gamma)\nu_0(s')$ is the initial contribution, and the second $\gamma\int P(s'|s,a)\pi(a|s)\nu^\pi(s)\,ds\,da$ is the transition contribution.

Here $\nu_0(s')$ is the probability $P(s_0=s')$ of initially being at $s'$, and is not the same function as $\nu^\pi$.

First term: $(1-\gamma)\nu_0(s')$

- Meaning: the weighted probability of being at $s'$ at time $t=0$.
- $\nu_0(s')$: the initial-state distribution, $P(s_0=s')$.
- $(1-\gamma)$: a normalization coefficient for time weights.
- Recall the definition of $\nu^\pi$: $\nu^\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t P(s_t=s)$.
- When $t=0$, $\gamma^0=1$.
- The contribution of time $t=0$ is therefore coefficient $(1-\gamma)$ multiplied by probability $\nu_0(s')$.

Second term: $\gamma\int P(s'|s,a)\pi(a|s)\nu^\pi(s)\,ds\,da$

- Meaning: the weighted probability of reaching $s'$ in one transition from any state $s$ at the previous time.
- $\nu^\pi(s)$: the probability density of state $s$ at the previous time.
- $\pi(a|s)$: the probability of choosing action $a$ in $s$.
- $P(s'|s,a)$: the probability of moving from $s$ to $s'$ after taking $a$.
- $\int \cdots dsda$: integrates over all possible “predecessor” states and actions (or sums $\sum$ in discrete spaces), collecting all possibilities leading to $s'$.
- $\gamma$: the crucial discount factor.
- Moving from $s$ to $s'$ takes one time step.
- If the previous time (say $t$) has weight $\gamma^t$, arrival at $s'$ occurs at $t+1$, with weight $\gamma^{t+1}$.
- The weight is discounted by a factor of $\gamma$ relative to the previous time, so multiplication by $\gamma$ is required.

## V. Theorems Concerning Occupancy Measures

Two further theorems follow.

Theorem 1: when an agent interacts with the same MDP under policies $\pi_1$ and $\pi_2$, the resulting occupancy measures $\rho^{\pi_1}$ and $\rho^{\pi_2}$ satisfy:

$$
\rho^{\pi_1} = \rho^{\pi_2} \Longleftrightarrow \pi_1 = \pi_2
$$

Theorem 2: given a valid occupancy measure $\rho$, the unique policy that generates it is:

$$
\pi_\rho = \frac{\rho(s,a)}{\sum_{a'}\rho(s,a')}
$$

## References

- [Hands-on Reinforcement Learning (translated title; in Chinese)](https://hrl.boyuai.com/).
