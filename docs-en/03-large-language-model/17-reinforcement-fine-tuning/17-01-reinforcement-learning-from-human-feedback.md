---
title: "17.1 Reinforcement Learning from Human Feedback"
chapter_title: "Reinforcement Fine-Tuning"
section_id: "17-01"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/17.强化微调/17.1 基于人类反馈的强化学习.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 17.1 Reinforcement Learning from Human Feedback

## I. Background

After supervised fine-tuning and reasoning-oriented reinforcement fine-tuning, a model is highly capable but has two problems:

1. Usability: it may mix Chinese and English, repeat itself, or speak aggressively, focusing only on solving the problem rather than making the answer comfortable for humans to read.

2. Safety: its values may not be aligned with humans.

## II. Core Principles

1. Reward model

Humans cannot score every training response. Instead, a model generates two or more answers to a question, humans rank their quality, and the answer–ranking data is used to train a reward model.

Let the reward model's parameters be $\phi$. For the same prompt $x$, consider two answers:

- $y_w$ (winner): the response human annotators prefer.
- $y_l$ (loser): the response human annotators consider worse.
- $r_\phi(x,y)$: the model's scalar score.

Define loss $L(\phi)$ as:

$$
L(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log\left(\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right)\right]
$$

Here, $\sigma$ is the Sigmoid function:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

It compresses values into $(0,1)$ to represent probabilities. $r_\phi(x,y_w)-r_\phi(x,y_l)$ is the score difference.

This applies cross-entropy to “the probability that the model correctly identifies the better answer,” encouraging higher scores for better answers and lower scores for worse ones.

The loss is fundamentally maximum-likelihood estimation of assigning a higher score to the preferred sample:

The Bradley–Terry model expresses this probability as:

$$
P(y_w>y_l)=\frac{\exp(r(y_w))}{\exp(r(y_w))+\exp(r(y_l))}
$$

Dividing numerator and denominator by $\exp(r(y_l))$ gives:

$$
P(y_w>y_l)=\frac{\exp(r(y_w)-r(y_l))}{1+\exp(r(y_w)-r(y_l))}=\sigma(r(y_w)-r(y_l))
$$

2. Reinforcement learning

Algorithms such as PPO and GRPO use the reward model from the second step to assign rewards to LLM answers and perform reinforcement learning. Taking PPO as an example:

In RLHF's PPO objective, a KL-divergence term is usually subtracted in addition to clipping. The actual KL divergence places an expectation outside the logarithm; the objective is the expectation of the entire $R_{\mathrm{total}}$:

$$
R_{total}=R_{model}(s,a)-\beta\log\frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}
$$

The first safeguard is the KL penalty, preventing loss of the model's foundations and uncontrolled deviation:

- Comparison: current policy $\pi_\theta$ versus SFT model $\pi_{ref}$.
- $\pi_{ref}$ is the anchor: it fixes the model's “teacher” or “factory settings” (the SFT model), remaining frozen throughout training.
- Role: a long-term constraint.
  - Regardless of how many epochs or updates occur, the model must not move too far from its factory settings.
  - Without this term, reward hacking may make the model output gibberish or exploit the reward model. The KL penalty constrains it to human-like language.

The second safeguard is PPO clipping, preventing destabilizing updates:

- Comparison: current policy $\pi_\theta$ versus previous policy $\pi_{\theta_{old}}$.
- $\pi_{\theta_{old}}$ is the anchor: the model's own state from a few minutes earlier. It changes dynamically.
- Role: a short-term constraint (step-size control).
  - It ensures mathematical stability of gradient updates. It does not guarantee closeness to the SFT model; it only ensures that the current parameter update is not so aggressive that it destabilizes the network.

Extension: the original PPO-Penalty algorithm versus penalties in practical RLHF

The “PPO with Adaptive KL Penalty” variant adds KL directly to the loss.

In RLHF, however, KL is usually injected through each step's reward rather than added directly to the final loss. This lets the value function V(s) perceive the cost. When predicting whether a state (generating a token) is good, V(s) automatically considers the cost of deviating from SFT if that cost is included in the reward. The agent can then anticipate through V(s): “If I begin producing nonsense, I might obtain a high score, but the severe KL penalty makes this a poor state.”

## References

- Ouyang, L., Wu, J., Jiang, X., et al. (2022). [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.
