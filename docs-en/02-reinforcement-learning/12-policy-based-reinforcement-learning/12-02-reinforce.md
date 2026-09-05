---
title: "12.2 REINFORCE"
chapter_title: "Policy-Based Reinforcement Learning"
section_id: "12-02"
language: en
source_language: zh
source_docx: "第2部分 强化学习/12.基于策略的强化学习/12.2 REINFORCE算法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 12.2 REINFORCE

## I. Transforming the Reward Term

The bracketed expression in the policy gradient formula can be viewed as a sum of $T+1$ terms. Each is “the gradient of log policy $\log\pi_\theta(a_t\mid s_t)$ at time $t$ with respect to $\theta$, multiplied by the full action sequence's reward $R(\tau)$.”

The gradient of $\log\pi_\theta(a_t\mid s_t)$ with respect to $\theta$ means “changing $\theta$ in this direction increases the probability of choosing $a_t$ in $s_t$ fastest.” If the action yields positive future rewards, its probability should rise; otherwise it should fall. Yet coefficient $R(\tau)=r_0+\cdots+r_T$ includes all rewards from 0 to $T$, which is clearly unreasonable: an action at $t=5$ cannot affect rewards already received from $t=0$ to $t=4$. Each gradient should therefore be multiplied by its future rewards. Replace $R(\tau)$ with:

Use return $G_t$ in place of the $Q$ function.

In the theorem, $Q^{\pi_\theta}(s_t,a_t)$ is the expected cumulative return after taking $a_t$ in $s_t$. REINFORCE does not know the true $Q$ function, but can calculate an observed cumulative return $G_t$ from a sampled trajectory as an approximation:

$$
G_t=r_t+\gamma r_{t+1}+\gamma^{2}r_{t+2}+\cdots+\gamma^{T-t}r_T
$$

$G_t$ is an unbiased estimate of $Q^{\pi_\theta}(s_t,a_t)$, but has high variance because it comes from a single sample.

The policy gradient becomes:

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\pi_\theta}
\bigl[
\sum_{t=0}^{T}
(\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'})
\nabla_\theta\log\pi_\theta(a_t|s_t)
\bigr]
$$

The outer sum from 0 to $T$ means that updating $\theta$ optimizes the policy at every time step in the action sequence. The inner sum from $t$ to $T$ means that each update's magnitude depends only on future rewards, not those received before $t$.

For each fixed $t$ and network, this takes an expectation over all $G_t$, so it can also be written:

$$
\nabla J(\theta)
=\mathbb{E}_{\tau\sim\pi}
\bigl[
\sum_{t=0}^{T}
\nabla_\theta\log\pi_\theta(a_t|s_t)Q^{\pi}(s_t,a_t)
\bigr]
$$

Here $Q_\pi(s_t,a_t)$ is the expected future cumulative reward $G_t$ for $(s_t,a_t)$.

In this update, the gradient of $\pi$ with respect to $\theta$ indicates “how to change $\theta$ to increase action probability $\pi$ for $a$ at $s$ fastest.” Taking $\log$ of $\pi$ “compresses” the dependent-variable direction without changing the steepest-ascent (gradient) direction, but maps probabilities in $(0,1)$ to $(-\infty,0)$, increasing gradient magnitude; probabilities farther from 1 yield larger gradients. When $G_t>0$, backpropagation tends to increase the probability of $a$; otherwise it tends to decrease it.

## II. Monte Carlo Approximation of the Expectation

The policy gradient theorem directly gives the gradient of cumulative reward $J$ with respect to $\theta$, weighting actions by policy $\pi$. Its formula is an expectation, however, requiring a weighted average over every possible trajectory, which cannot be computed in practice. Monte Carlo (estimating probabilities by frequencies) approximates it: run the current policy to sample $N$ complete trajectories and compute the cumulative future reward $G_t$ at each time step $t$. This is equivalent to estimating $Q_\pi(s_t,a_t)$ with $G_t$.

Use these trajectories as samples to approximate the expectation, similarly to minibatch stochastic gradient descent. Each explored action path contributes:

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}\sum_{i=1}^{N}
\bigl(
\sum_{t=0}^{T}
\nabla_\theta\log\pi_\theta(a_t^{(i)}|s_t^{(i)})G_t^{(i)}
\bigr)
$$

Monte Carlo estimates have high variance, mainly from unstable $G_t$. In detail:

$$
\text{Gradient}\approx\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot G_t
$$

Term A: $\nabla_\theta\log\pi_\theta(a_t|s_t)$, “our own part”

This asks: “In which direction should $\theta$ change to increase my probability of action $a_t$ in state $s_t$?”

- Deterministic given the sample: once $(s_t,a_t)$ is sampled, this is purely a differentiation problem. $\pi_\theta$ is our designed neural network (such as an MLP), with a fully known mathematical form. Backpropagation calculates the gradient vector exactly.

Term B: $G_t$, “the environment and future part”

This asks: “After taking $a_t$ in $s_t$, how many points did I actually receive before the game ended?”

- Highly stochastic: even with exactly the same $s_t$ and $a_t$, the next $G_t$ may differ completely. $G_t$ includes all randomness from $t$ through $T$:
  1. Environmental randomness: wind may blow left this time and right next time (transition probability $P(s'|s,a)$).
  2. Policy randomness: action A may be selected at $t+1$ this time and B next time (randomness of future policy $\pi$).
- Accumulated: $G_t=r_t+r_{t+1}+r_{t+2}+\cdots$. Every $r$ is random. The variance of a sum of independent (or correlated) random variables usually grows linearly or faster with their number. Long trajectories accumulate noise until it explodes.

## III. REINFORCE with a Baseline

The preceding derivation assumes positive rewards increase probabilities. Often this merely makes “all updated probabilities tend to increase,” which is not what we need. Actions should be chosen by their value relative to others, not their absolute value.

Although REINFORCE improves on simple policy gradients, estimating gradients from sampled trajectories often has high variance. Successive sampled gradients can differ greatly, affecting stability and convergence speed.

Introduce a baseline to address this. The core idea is to subtract a predicted value, reducing variance without changing the expectation. Use $G_t-b(s_t)$ instead of $G_t$ as the weight, where $b(s_t)$ is the baseline for state $s_t$. To preserve the expectation, this baseline should be independent of policy parameters $\theta$.

Intuitively, encourage an action when $G_t$ exceeds $b(s_t)$ and suppress it when $G_t$ is below $b(s_t)$.

REINFORCE with a baseline computes gradients as:

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\tau\sim\pi_\theta}
\bigl[
\sum_{t=0}^{T}
(G_t-b(s_t))\nabla_\theta\log\pi_\theta(a_t|s_t)
\bigr]
$$

The only difference from REINFORCE is using $G_t-b(s_t)$ rather than $G_t$.

Introducing baseline $b(s_t)$ gives:

$$
\nabla J(\theta)
=\mathbb{E}
\bigl[
\sum_{t=0}^{T}
\nabla\log\pi(a_t|s_t)\cdot(G_t-b(s_t))
\bigr]
$$

Usually $b(s_t)$ is the current state's average value, $V^{\pi}(s_t)$. Returning to the example, suppose the average score is $+55$:

- Poor performance ($+10$): $10-55=-45$. (The gradient reverses, directly suppressing the action.)
- Good performance ($+100$): $100-55=+45$. (The gradient is positive, directly rewarding the action.)

Since $b(s_t)$ is independent of policy parameters $\theta$, it does not affect optimization. Proof:

The key is that $\mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\cdot b(s_t)]$ equals zero, because:

$$
\begin{aligned}
\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t \mid s_t)b(s_t)]
&= b(s_t)\sum_a \pi_\theta(a \mid s_t)\nabla_\theta \log \pi_\theta(a \mid s_t) \\
&= b(s_t)\nabla_\theta(\sum_a \pi_\theta(a \mid s_t)) \\
&= b(s_t)\nabla_\theta 1 \\
&= 0
\end{aligned}
$$

Thus, provided $b(s_t)$ does not depend on action $a_t$ and is a function of state, it has no systematic effect on the gradient estimate in the left-hand expectation, introducing no bias and preserving unbiasedness.

## IV. Summary

Policy-based RL does not enumerate all policies and is more flexible than value-based RL. However, it can become trapped in local optima (finding an adequate policy and not seeking a better one, which can be avoided by appropriately increasing initial policy randomness). Parameters change smoothly rather than abruptly as value-based action choices can after small changes in $Q$. To ensure accurate Monte Carlo estimates of $G_t$, it requires more data than value-based RL.

## References

- Williams, R. J. (1992). [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696). Machine Learning.
