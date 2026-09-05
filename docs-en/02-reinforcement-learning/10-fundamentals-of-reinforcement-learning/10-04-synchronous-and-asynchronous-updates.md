---
title: "10.4 Synchronous and Asynchronous Updates"
chapter_title: "Fundamentals of Reinforcement Learning"
section_id: "10-04"
language: en
source_language: zh
source_docx: "第2部分 强化学习/10.强化学习的基本知识/10.4 同步更新和异步更新.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 10.4 Synchronous and Asynchronous Updates

## I. Synchronous Updates

Large models such as ChatGPT and Llama are trained using Proximal Policy Optimization (PPO) or its variants. Asynchronous updates are almost nonexistent in this setting.

Process:

1. Rollout data collection: multiple GPU workers generate text with the LLM in parallel.
2. Synchronization barrier: the system must wait for every GPU to generate the specified amount of data (for example, 64 samples per card).
3. Backward gradient computation: compute gradients from the aggregated data.
4. Gradient averaging through All-Reduce: use distributed training such as Distributed Data Parallel (DDP) to average gradients across all GPUs.
5. Update: all GPUs update their model parameters simultaneously.

## II. Asynchronous Updates

In some settings, however, dozens of worker threads run simultaneously at vastly different speeds. If thread A finishes and waits for thread B, substantial resources are wasted. Its output is therefore often sent directly to a central server, where the model is updated as soon as enough data arrive, without waiting for B. Examples include:

1. AutoML and neural architecture search (NAS): a typical setting

- Scenario: AI designs neural network architectures or tunes hyperparameters.
- Action A (fast): train a shallow three-layer network (30 seconds).
- Action B (slow): train a deep 100-layer ResNet (two hours).

2. AI for Systems (database/compiler/system optimization)

This is currently a popular industrial RL application area, with extremely high variance in action duration.

- Scenario A: database parameter tuning
- Action: modify the database configuration and run a benchmark.
- Difference: changing `work_mem` may require only a restart (fast); changing an `index` strategy may require rebuilding indexes (extremely slow, tens of minutes).
- Scenario B: compiler optimization
- Action: decide which optimization pass to apply to a code block.
- Difference: “dead-code elimination” is instantaneous, whereas “polyhedral optimization” or “loop unrolling” may greatly increase compilation time.

A3C is a classic asynchronous actor–critic algorithm. OpenAI Baselines subsequently introduced the synchronous A2C variant and reported comparable performance to A3C with the same hardware and hyperparameters. Their relative merits therefore depend on implementation, hardware, and task; one cannot categorically claim that A3C is always inferior to A2C.

## III. Time Weighting for Asynchronous Updates

1. Problems with asynchronous updates

Asynchronous updates are mainly used where action durations differ greatly. In standard distributed RL frameworks, multiple actors simultaneously generate and execute different code solutions, then send results (code, execution outcomes, and rewards) to a learner for model updates. This causes problems in practice: in 10 minutes (600 seconds), an actor may execute action A 120 times but action B only twice. The learner receives 120 gradient-update signals for A but only two for B. Model parameters are “flooded” by A's gradients, making the agent mistakenly believe A is the best policy and trapping it in a local optimum.

Consider the two settings above. When AI chooses neural architectures without additional adjustments, it generates shallow networks at a frantic pace because they produce many gradient updates quickly. Although deep networks perform well, slow feedback causes AI to “forget” them. System optimization similarly favors shorter actions.

The mathematical explanation is:

The gradient update is usually:

$$
\nabla J(\theta)=\mathbb{E}_{a\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a|s)\cdot A(s,a)\right]
$$

Here $\pi_\theta$ is the policy distribution and $A(s,a)$ the advantage function. Note that this expectation assumes sampling according to $\pi_\theta$.

In an asynchronous distributed system, data-generation rate is inversely proportional to execution time $\Delta t$. The actual sampling probability $P_{sample}(a)$ is no longer simply policy probability $\pi(a)$; time distorts it:

$$
P_{sample}(a)\propto \frac{\pi(a)}{\Delta t(a)}
$$

This artificially amplifies the probability of sampling shorter actions. Updating directly with the collected data therefore optimizes a time-distorted objective rather than the original one.
In fact, off-policy algorithms with replay buffers (DQN, SAC, DDPG) are theoretically also nonsynchronous: workers simply add data to the buffer (potentially asynchronously), and the learner randomly samples data for training. Asynchronous workers add fast actions more quickly, increasing their representation in the buffer. However, since the learner samples uniformly at random, if the buffer is large and well mixed, the bias is diluted, though not eliminated. When action-speed differences are small, this error is generally ignored.

2. Solution: time weighting

Since short durations cause more frequent sampling, weight each update according to its duration.

One correction for duration bias introduces time $\Delta t_k$ (the execution duration of action $k$) as a multiplicative weight. The corrected gradient estimate $\hat{g}$ is:

$$
\hat{g}\approx \Delta t_k\cdot \nabla_\theta\log\pi_\theta(a_k|s_k)\cdot A(s_k,a_k)
$$

$$
\begin{aligned}
\mathbb{E}_{sample}[\hat{g}]
&= \sum_a P_{sample}(a)\cdot\left(\Delta t(a)\cdot\nabla\log\pi(a)\cdot A(a)\right)\\
&\propto \sum_a \left(\frac{\pi(a)}{\Delta t(a)}\right)\cdot\Delta t(a)\cdot\nabla\log\pi(a)\cdot A(a)\\
&= \sum_a \pi(a)\cdot\nabla\log\pi(a)\cdot A(a)\\
&= \mathbb{E}_{a\sim\pi}\left[\nabla\log\pi(a)\cdot A(a)\right]
\end{aligned}
$$

Conclusion: multiplying by $\Delta t(a)$ exactly cancels the denominator's $1/\Delta t(a)$ sampling-frequency factor. The expected gradient returns to the unbiased policy distribution $\pi(a)$, giving long-duration actions “fair consideration.”
Two practical examples:

(1) Quantitative trading

In finance, RL generates trading policies.

- Scenario: an agent decides whether to open, close, or hold a position.
- Action differences:
  - High-frequency scalping: close after holding for seconds to earn tiny spreads. Trades are extremely frequent and data points extremely dense.
  - Trend following: hold for days or weeks while waiting for major moves.
- Consequence: in asynchronous frameworks, high-frequency trading produces thousands or tens of thousands of times more samples than trend trading. The learner is overwhelmed by high-frequency noise, making it believe that “only frequent trading makes money” and miss major long-term opportunities.
- Correction: assign very large weights ($\Delta t$) to long-held trades, telling the model, “This trade took a week but earned 10%; it is worth learning carefully.”

(2) Drug discovery and molecular simulation

- Scenario: an agent generates molecular structures and calls physical simulators to verify their properties.
- Action differences:
  - Low-precision estimation: empirical formulas or coarse surrogate models (seconds).
  - High-precision simulation: density functional theory (DFT) or all-atom dynamics simulations (hours or days).
- Consequence: AI favors molecules that “low-precision simulators can easily calculate,” avoiding structurally complex, potentially exceptional drugs requiring high-precision computation for verification.

Note: for numerical stability, $\Delta t$ is usually clipped or normalized to prevent extremely long tasks from causing exploding gradients.

## References

- [Hands-on Reinforcement Learning (translated title; in Chinese)](https://hrl.boyuai.com/).
- Mnih, V., Badia, A. P., Mirza, M., et al. (2016). [Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html). ICML 2016.
- Dhariwal, P., et al. (2017). [OpenAI Baselines: ACKTR & A2C](https://openai.com/index/openai-baselines-acktr-a2c/). OpenAI.
- Yang, S., He-Yueya, J., & Liang, P. (2025). [Reinforcement Learning for Machine Learning Engineering Agents](https://arxiv.org/abs/2509.01684). arXiv:2509.01684.
