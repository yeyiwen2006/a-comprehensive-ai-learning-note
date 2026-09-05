---
title: "12.1 Basic Ideas of Policy-Based Reinforcement Learning"
chapter_title: "Policy-Based Reinforcement Learning"
section_id: "12-01"
language: en
source_language: zh
source_docx: "第2部分 强化学习/12.基于策略的强化学习/12.1 基于策略的强化学习的基本思想.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 12.1 Basic Ideas of Policy-Based Reinforcement Learning

## I. Limitations of Value-Based Methods

Value-based reinforcement learning relies on learning Q(s,a), the value of an action in a given state. This requires enumerating all X actions to select the best, with discrete outputs. For continuous actions (such as robotic arm control), a network can approximate at most as many discrete actions as it has output nodes, producing stiff motion. Otherwise, fitting Q requires comparing expectations for almost infinitely many policies to select the optimum, requiring almost infinitely many parameters. We therefore introduce policy-based reinforcement learning.

## II. Policy-Based Reinforcement Learning

1. Overall idea

Directly learn the probability distribution of actions in each state and sample accordingly. As in learning to ride a bicycle (or precise continuous robot movements), the best expected state value cannot be calculated; instead, let the agent continually select policies in each state (more precisely, policy probability distributions: a pathfinding puppy might learn to turn left with 70% probability and right with 30% at an intersection). Each complete journey yields cumulative reward G (unlike the Q function's expectation under the optimal policy). The ultimate objective is a policy π maximizing expected cumulative reward J. When policy is represented by a parameter vector θ (such as neural network weights), the problem becomes finding optimal θ.

2. Neural network function

The network takes state s and outputs action values or action-related parameters. It fits π, the probability distribution of the action-value vector or action-related parameters given the state (a density for continuous values). It is experience-based, with experience stored in network parameters θ.

3. Objective function (loss function)

Let the initial state be fixed at s0. The objective is to maximize expected cumulative reward J=E_τ~π(R(τ)) for a complete action sequence. R(τ) is the reward received over the full sequence τ, and the expectation uses the trajectory distribution induced by π.

4. Policy gradient

The gradient of expected cumulative reward J (the expectation of R(τ)) with respect to parameter vector θ:

Gradient expression: start from the objective definition and transform it using the integral form of expectation and the log-likelihood trick.

$$
\nabla_\theta J(\theta)
=\nabla_\theta\int P(\tau|\theta)R(\tau)d\tau
=\int\nabla_\theta P(\tau|\theta)R(\tau)d\tau
$$

Using $\nabla_\theta P(\tau|\theta)=P(\tau|\theta)\nabla_\theta\log P(\tau|\theta)$ gives:

$$
\nabla_\theta J(\theta)
=\int P(\tau|\theta)R(\tau)\nabla_\theta\log P(\tau|\theta)d\tau
=\mathbb{E}_{\tau\sim\pi_\theta}\left[R(\tau)\nabla_\theta\log P(\tau|\theta)\right]
$$

We have transformed an integral of a gradient into an expectation, allowing the gradient to be approximated by sampling.

Decompose action sequence τ into the “elementary actions” at each time step:

Factorizing trajectory probability: the probability of trajectory $\tau=(s_0,a_0,s_1,a_1,\cdots)$ is:

$$
P(\tau|\theta)=p(s_0)\prod_{t=0}^{T}\pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)
$$

Only the policy term $\pi_\theta(a_t|s_t)$ depends on $\theta$. Taking the log gradient therefore eliminates environmental dynamics (initial distribution $p(s_0)$ and transitions $P(s_{t+1}|s_t,a_t)$):

$$
\nabla_\theta\log P(\tau|\theta)=\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

(Here s_i is the state and a_i the action, corresponding to a product of (action-selection probability * state-transition probability).)

Final form: substituting into the expectation yields the classic policy gradient theorem:

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{\tau\sim\pi_\theta}
\left[
R(\tau)\cdot\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)
\right]
$$

What is $\nabla_\theta\log\pi_\theta(a_t|s_t)$?

- A vector indicating “how to move in parameter space to increase the probability of choosing $a_t$ in $s_t$.”
- Simply put: it seeks to make this action more probable.

This term determines the gradient direction, the update direction for vector θ. R determines the sign and magnitude (scaling the gradient vector): larger positive rewards encourage gradient ascent in that direction, while negative rewards do the opposite. The gradient is an expectation, so sampling under π_θ yields an unbiased estimate.

Since the expectation is indexed by π_θ, the objective changes with the policy, and the updated policy relates directly to the behavior policy. This is an online learning algorithm.

## References

- Williams, R. J. (1992). [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696). Machine Learning, 8, 229-256.
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html). NeurIPS 1999.
- Sutton, R. S., & Barto, A. G. (2018). [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). MIT Press.
