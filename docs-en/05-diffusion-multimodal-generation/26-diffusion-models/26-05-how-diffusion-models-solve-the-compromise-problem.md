---
title: "26.5 How Diffusion Models Solve the “Compromise Problem”"
chapter_title: "Diffusion Models"
section_id: "26-05"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/26.扩散模型/26.5 扩散模型如何解决“折中问题”.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 26.5 How Diffusion Models Solve the “Compromise Problem”

## I. The Compromise Problem

Suppose our action space is one-dimensional (for example, the steering-wheel angle). According to expert driving data, when a tree is ahead, the probability of swerving left (corresponding to $x=-1$) is 50%, and the probability of swerving right (corresponding to $x=1$) is 50%. The probability of driving straight ahead (corresponding to $x=0$, or hitting the tree) is 0%.

Under this definition, the true probability density function $p(x)$ is a **bimodal distribution**:

- At $x=-1$ and $x=1$, $p(x)$ reaches local maxima (peaks).
- At $x=0$, $p(x)\approx 0$ (a deep valley or saddle point).

If you train a neural network $f_\theta(s)$ to output the action $x$ directly and use an MSE loss:

$$
Loss=\mathbb{E}_{x\sim p(x)}\left[\lVert f_\theta(s)-x\rVert^2\right]
$$

Minimizing this loss is then equivalent to maximizing the joint probability density of all training data. In this problem, the optimal solution is to output the mathematical expectation of the entire distribution:

$$
\hat{x}=\int x\cdot p(x)dx=0.5\times(-1)+0.5\times(1)=0
$$

Although the true probability density $p(0)$ at $x=0$ is 0, MSE forces the output to this “compromise point.” **This is because MSE fits the distribution's mean and does not care about the shape of the probability density itself.**
The same issue occurs in image generation: it can easily average pixels from different images and generate something that resembles none of them.

## II. How Do Diffusion Models Avoid This “Compromise”?

A diffusion model learns the gradient of the log probability density, namely the score function $\nabla_x\log p(x)$.

Imagine standing on the bimodal terrain of $p(x)$. Langevin dynamics proceeds as follows:

[Algorithmic workflow for multimodal sampling with Langevin dynamics]

- **Step 1: Random initialization.** Sample an initial point $x_T$ from Gaussian noise. Suppose we are extremely unlucky and the point lands exactly at $x=0$ (the tree-hitting mean point). At this saddle point, the theoretical gradient is $\nabla_x\log p(0)=0$.
- **Step 2: Inject noise (break the balance).** The model now adds random noise $\sqrt{\Delta t}z$. The point originally at $x=0$ receives a random push and becomes $x=0.01$ (slightly to the right).
- **Step 3: Gradient guidance (climb uphill).** Once the point leaves the exact central saddle point, the attraction of the right-hand peak begins to appear at $x=0.01$. $\nabla_x\log p(0.01)$ produces a strong gradient vector **to the right (in the positive direction)**.
- **Step 4: Iterative collapse.** In subsequent iterations, the gradient continues to point uphill (to the right). Although noise is still injected, its variance becomes progressively smaller because of annealing. The point quickly “climbs” along the gradient to the peak at $x=1$ and stabilizes there.

**Conclusion**: Langevin dynamics searches for local maxima (modes) on the probability-density surface. Because $x=0$ is a low-density valley (or saddle point), the gradient pushes the point away from the mean, eventually causing it to collapse randomly (depending on the initial and injected noise) to a definite peak on either the left (corresponding to $x=-1$) or the right (corresponding to $x=1$). **It will never remain at the extremely low-density compromise point.**

## III. Can Autoregressive Models Handle Such a Distribution?

Yes, but they are only suitable for discretized distributions.

1. **Autoregressive models in “discrete spaces” (such as language generation)**

When GPT predicts the next word, it outputs a probability distribution over the entire vocabulary (through softmax and cross-entropy loss).

If the correct next word is “left” or “right,” the model assigns a probability of 0.5 to each. During inference, **sampling or greedy search** necessarily selects one of the two tokens, “left” or “right.” It never outputs a nonexistent “in-between word,” because the vocabulary is discrete. Thus, AR models solve the multimodality problem perfectly in discrete spaces.

2. **The difficulty of autoregressive models in “high-dimensional continuous spaces” (such as embodied actions or pixels)**

The action space of embodied AI (such as a robotic arm's joint-torque vector $[0.12, -0.45, 1.23, \ldots]$) or the pixel space of a high-resolution image is continuous and high-dimensional.

- **If AR uses MSE regression**: it immediately degenerates into the “compromise mean (hitting the tree)” described earlier and cannot express multimodality.
- **If the continuous space is forcibly discretized before using AR**: this resembles VQ-VAE turning images into discrete tokens followed by autoregressive generation with a Transformer (such as early VideoPoet or Parti). This approach avoids the compromise, but introduces an extremely large computational burden and the **exposure bias** mentioned above, causing errors to multiply during long-sequence generation.

## References

- Song, J., Meng, C., & Ermon, S. (2021). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). ICLR.
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364). NeurIPS.
