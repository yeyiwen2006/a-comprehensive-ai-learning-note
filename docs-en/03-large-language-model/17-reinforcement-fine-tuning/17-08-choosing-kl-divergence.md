---
title: "17.8 Choosing KL Divergence"
chapter_title: "Reinforcement Fine-Tuning"
section_id: "17-08"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/17.强化微调/17.8 KL散度的选择.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 17.8 Choosing KL Divergence

## I. KL Divergence in LLMs

In large language models, KL divergence measures the difference between output probability distributions. It is calculated over a vocabulary of size $V$, rather than over the hidden-layer dimensions before the output, because the outputs of large language models are fundamentally discrete tokens.

$D(P \Vert Q)=\sum_{i=1}^{V}p_i\log\frac{p_i}{q_i}$ is equivalent to the expectation of $\log(p_i/q_i)$ under the distribution $p$.

For an LLM, the KL divergence for an autoregressive output of $n$ tokens is:

$$
D_{KL}(P(Y) \Vert Q(Y))=\sum_Y P(Y)\log\frac{P(Y)}{Q(Y)}
$$

Because directly summing over all possible sequences is infeasible (there are $V^n$ possibilities), we need to use the autoregressive property to decompose it:

$$
\begin{aligned}
D_{KL}(P(Y) \Vert Q(Y))
&= \mathbb{E}_{Y\sim P}\left[\sum_{t=1}^{n}\left(\log P(y_t\mid y_{<t})-\log Q(y_t\mid y_{<t})\right)\right] \\
&= \sum_{t=1}^{n}\mathbb{E}_{Y\sim P}\left[\log\frac{P(y_t\mid y_{<t})}{Q(y_t\mid y_{<t})}\right]
\end{aligned}
$$

The difficulty here lies in $Y\sim P$, which requires considering the probability distribution of the entire sequence. Notice that the conditional probabilities on the right are conditioned on $y_{<t}$. Given a known distribution of $y_{<t}$, the autoregressive property also determines the distribution of $y_t$, so the distribution of $Y=\{y_1,\ldots,y_t\}$ is known as well. We can therefore rewrite the expression as:

$$
D_{KL}(P(Y) \Vert Q(Y))
=\sum_{t=1}^{n}\mathbb{E}_{y_{<t}\sim P(y_{<t})}\left[
D_{KL}\left(P(y_t\mid y_{<t}) \Vert Q(y_t\mid y_{<t})\right)
\right]
$$

This means that the KL divergence between two sequence distributions equals the sum of the expected KL divergences between their conditional probability distributions at each step (where the expectation is calculated over histories generated from distribution P). Using Monte Carlo estimation, averaging over trajectories generated under policy P approximates the expectation under policy P.

## II. The K1 and K3 Estimators

For a sequence of length T:

**The Naïve K1 Estimator**

This is the most intuitive Monte Carlo estimate: it directly calculates the log-likelihood ratio:

$$
K1=\sum_{t=1}^{T}K1_t
=\sum_{t=1}^{T}\log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}
$$

Properties: it is an unbiased estimate of $D_{KL}(\pi_\theta \Vert \pi_{\mathrm{ref}})$, but has relatively high variance in practice.

**The Schulman K3 Estimator**

Proposed by John Schulman to reduce variance, it uses the property $\log(x)\le x-1$ for approximation:

$$
K3=\sum_{t=1}^{T}K3_t
=\sum_{t=1}^{T}\left(
\frac{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}{\pi_\theta(y_t\mid x,y_{<t})}
-1
-\log\frac{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}{\pi_\theta(y_t\mid x,y_{<t})}
\right)
$$

Properties: it is also an unbiased estimate of $D_{KL}(\pi_\theta \Vert \pi_{\mathrm{ref}})$ (its expectation is the KL divergence), and has lower variance than K1.

The KL divergence term can be subtracted from the reward or added to the loss. In actual training, K1 in the loss and K3 in the reward are unstable; K3 in the loss (lower variance) or K1 in the reward (unbiased gradient) is generally used. GRPO uses the former by default, but research in 2025 showed that the latter may actually perform better, because the former is itself unbiased but has a biased gradient.

### (III) Proving That K1 in the Reward Has an Unbiased Gradient and K3 in the Loss Has a Biased Gradient

The gradient of the true objective we need to optimize (reverse KL) is:

$$
\nabla_\theta D_{KL}(\pi_\theta \Vert \pi_{\mathrm{ref}})
=\mathbb{E}_{y\sim\pi_\theta}\left[
\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\nabla_\theta\log\pi_\theta
\right]
$$

Note: to simplify the notation, the condition $x$ and summation signs are omitted so that we can focus on the core terms.

**Derivation 1: K1 in the Reward (Unbiased)**

When K1 is placed in the reward, we do not differentiate K1 itself (it is treated as a constant reward). Instead, we use the properties of REINFORCE/PPO:

$$
\mathrm{Gradient}
=\mathbb{E}_{y\sim\pi_\theta}\left[K1\cdot\nabla_\theta\log\pi_\theta\right]
$$

Substituting $K1=\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$:

$$
\mathrm{Gradient}
=\mathbb{E}_{y\sim\pi_\theta}\left[
\left(\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right)
\nabla_\theta\log\pi_\theta
\right]
$$

Conclusion: this is identical to the expression for the true reverse KL gradient, so it is unbiased.

**Derivation 2: K3 in the Loss (Biased)**

When K3 is placed in the loss, we directly calculate the expectation of $\nabla_\theta K3$. Let the ratio be $r=\frac{\pi_{\mathrm{ref}}}{\pi_\theta}$; then $K3=r-1-\log r$. Note that $\log r=\log\pi_{\mathrm{ref}}-\log\pi_\theta$.

We need to calculate $\mathbb{E}_{y\sim\pi_\theta}[\nabla_\theta K3]$. First, consider the terms in $\nabla_\theta K3$:

1. $\nabla_\theta r=\nabla_\theta(\pi_{\mathrm{ref}}\cdot\pi_\theta^{-1})=\pi_{\mathrm{ref}}\cdot(-1)\pi_\theta^{-2}\cdot\nabla_\theta\pi_\theta=-\frac{\pi_{\mathrm{ref}}}{\pi_\theta}\cdot\frac{\nabla_\theta\pi_\theta}{\pi_\theta}=-r\nabla_\theta\log\pi_\theta$
2. $\nabla_\theta(-1)=0$
3. $\nabla_\theta(-\log r)=\nabla_\theta(\log\pi_\theta-\log\pi_{\mathrm{ref}})=\nabla_\theta\log\pi_\theta$ (assuming $\pi_{\mathrm{ref}}$ is fixed)

Combining these terms gives $\nabla_\theta K3$:

$$
\nabla_\theta K3
=-r\nabla_\theta\log\pi_\theta+\nabla_\theta\log\pi_\theta
=(1-r)\nabla_\theta\log\pi_\theta
$$

Now calculate its expectation:

$$
\mathbb{E}_{y\sim\pi_\theta}[\nabla_\theta K3]
=\mathbb{E}_{y\sim\pi_\theta}\left[(1-r)\nabla_\theta\log\pi_\theta\right]
$$

$$
\begin{aligned}
&=\mathbb{E}_{y\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta\right]
-\mathbb{E}_{y\sim\pi_\theta}\left[
\frac{\pi_{\mathrm{ref}}}{\pi_\theta}\nabla_\theta\log\pi_\theta
\right] \\
&=-\mathbb{E}_{y\sim\pi_\theta}\left[
\frac{\pi_{\mathrm{ref}}}{\pi_\theta}\nabla_\theta\log\pi_\theta
\right]
\end{aligned}
$$

The first term is $0$ (the score function property).

Comparison with the true gradient:

- True reverse KL gradient: $\mathbb{E}\left[\log\left(\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right)\nabla\log\pi_\theta\right]$
- K3-in-loss gradient: $\mathbb{E}\left[-\frac{\pi_{\mathrm{ref}}}{\pi_\theta}\nabla\log\pi_\theta\right]$

In fact, the gradient of the K3 estimator becomes the gradient of the corresponding forward KL divergence, rather than reverse KL divergence. This makes model training more inclined to cover existing modes than to explore the regions with the highest rewards within those modes.

The score function property is explained below:

**1. Why: Mathematical Derivation**

We want to prove:

$$
\mathbb{E}_{y\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(y)\right]=0
$$

The proof proceeds as follows:

**Step 1: Expand the definition of the expectation.** The expectation $\mathbb{E}$ is a weighted sum over all possible $y$ (a sum for discrete LLM token sequences, and an integral for a continuous space).

$$
\mathbb{E}_{y\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(y)\right]
=\sum_y \pi_\theta(y)\cdot\nabla_\theta\log\pi_\theta(y)
$$

**Step 2: Apply the log-derivative trick.** By the chain rule in calculus, $\nabla\log f(x)=\frac{\nabla f(x)}{f(x)}$. Applying it to $\log\pi_\theta(y)$:

$$
\nabla_\theta\log\pi_\theta(y)
=\frac{\nabla_\theta\pi_\theta(y)}{\pi_\theta(y)}
$$

**Step 3: Substitute back and cancel.** Substitute the result of Step 2 into the expression in Step 1:

$$
\sum_y \pi_\theta(y)\cdot
\frac{\nabla_\theta\pi_\theta(y)}{\pi_\theta(y)}
=\sum_y \nabla_\theta\pi_\theta(y)
$$

**Step 4: Exchange the order of summation and differentiation.** Under certain regularity conditions, we can exchange the summation sign $\sum$ and gradient operator $\nabla_\theta$:

$$
\sum_y \nabla_\theta\pi_\theta(y)
=\nabla_\theta\left(\sum_y \pi_\theta(y)\right)
$$

**Step 5: Use probability normalization.** This is the crucial step. For any probability distribution, the probabilities of all possible events must always sum to 1:

$$
\sum_y \pi_\theta(y)=1
$$

Therefore:

$$
\nabla_\theta\left(\sum_y \pi_\theta(y)\right)
=\nabla_\theta(1)=0
$$

Hence:

$$
\mathbb{E}_{y\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(y)\right]=0
$$

**2. Intuitive Understanding**

"Conservation of probability mass": the gradient $\nabla_\theta\pi_\theta(y)$ represents whether the probability of sample $y$ increases or decreases when we make a small adjustment to the parameters $\theta$.

- If we increase the probabilities of some samples (positive gradients), then the probabilities of other samples must decrease (negative gradients) so that the total probability always remains 1.
- The expression $\mathbb{E}[\nabla\log\pi]=\sum\nabla\pi$ actually calculates the sum of the probability changes across all possible samples.
- Because the total probability (the size of this "cake") is fixed at 1, all the changes must sum to 0.

## References

- Schulman, J. (2020). [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html).
- Liu, K., Liu, J. K., Chen, M., & Liu, Y. (2025). [Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization](https://arxiv.org/abs/2510.01555). arXiv:2510.01555.
- Ziegler, D. M., Stiennon, N., Wu, J., et al. (2019). [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593). arXiv:1909.08593.
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.
- Ahmadian, A., Cremer, C., Gallé, M., et al. (2024). [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740). ACL 2024.
