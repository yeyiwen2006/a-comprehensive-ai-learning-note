---
title: "28.6 Different Objectives for Video Diffusion Models"
chapter_title: "Multimodal Generation"
section_id: "28-06"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/28.多模态生成/28.6 视频生成扩散模型的不同目标函数.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 28.6 Different Objectives for Video Diffusion Models

## I. Shortcut Forcing: Composing Step Sizes in Flow-Matching Models

### (1) Background

In video generation or dynamics modeling, conventional diffusion or flow-matching models usually require dozens of denoising iterations, such as $K=64$, to generate a high-quality next state $x_{t+1}$. This is unacceptable for RL interaction requiring extremely fast inference.

Dreamer 4 uses Shortcut Forcing to improve large-step performance and reduce the required number of steps.

### (2) Core idea

During training, the model randomly samples a step size $d$, for example $d\in\{1/64,2/64,4/64,\ldots,1\}$. Different losses are selected according to $d$, effectively building a bootstrapped distillation ladder from small exploratory steps to large jumps.

For the smallest step $d=1/64$, the model needs only a small denoising update, so the flow-matching loss $\mathcal{L}_{\mathrm{flow}}$ is used. Most conventional diffusion models use $v$-prediction, predicting the velocity direction:

$$
v=\epsilon-x
$$

Here, $x$ is clean data and $\epsilon$ Gaussian noise. From a frequency-domain perspective based on the Fourier transform, the target includes white noise. Natural-image spectra approximately follow a $1/f$ law, with greater low-frequency energy, while white noise has a flat spectrum. Forcing the network to learn $v$ with substantial high-frequency content can lead to error accumulation and high-frequency artifacts in long-video generation.

Shortcut Forcing uses $x$-prediction, directly predicting clean latent representations. At any noise time $\tau\in[0,1]$, the network attempts to see through the noise and output the final noise-free state. To keep predictions consistent across step sizes, it introduces a bootstrap loss: network outputs are first converted back to $v$-space, then the bootstrap loss is scaled to $x$-space using $(1-\tau)^2$ for optimization.

Dreamer 4 forces the model to predict the final clean representation directly at any noise level $\tau\in[0,1]$:

$$
\hat{x}_1=f_\theta(x_\tau,\tau,c)
$$

The base flow-matching loss computes MSE directly in $x$-space:

$$
\mathcal{L}_{\mathrm{flow}}=\left\|f_\theta(x_\tau,\tau,c)-x\right\|_2^2
$$

For a larger sampled step $d=1/8$, the model should replace several steps with one. Fitting real data across a large step directly can produce blur or physically inconsistent artifacts. Training therefore uses the bootstrapped consistency loss $\mathcal{L}_{\mathrm{bootstrap}}$:

1. First let the model take two smaller half steps with its current capability, twice $d/2=1/16$.
2. Use the accumulated result of these half steps as a temporary teacher target.
3. Then let the model take one full step, $d=1/8$.
4. Compute $\mathcal{L}_{\mathrm{bootstrap}}$, forcing the one-large-step output to equal the two-small-step output.

Because computing cross-step losses directly in $x$-space has inconsistent scales, Dreamer 4 converts predicted $\hat{x}_1$ back to $v$-space:

$$
\hat{v}_\tau=\frac{\hat{x}_1-x_\tau}{1-\tau}
$$

After computing the MSE, multiply by $(1-\tau)^2$ to restore the $x$-space scale:

$$
\mathcal{L}_{\mathrm{bootstrap}}=(1-\tau)^2\left\|\hat{v}_\tau-v_{\mathrm{target}}\right\|_2^2
$$

To concentrate limited parameter capacity on the most informative denoising phase, near clean data, Dreamer 4 introduces a weight that increases linearly with signal level:

$$
w(\tau)=0.9\tau+0.1
$$

The overall loss selects $\mathcal{L}_{\mathrm{flow}}$ or $\mathcal{L}_{\mathrm{bootstrap}}$ according to sampled step size $d$, then multiplies it by $w(\tau)$ for optimization.

## II. The EDM Formulation: A Gradually Changing Diffusion Objective

### (1) Background

Standard DDPM trains the neural network to predict the noise $\epsilon$ added to clean images. At the beginning of generation, time step $T$ with very high noise, the network input is almost pure Gaussian noise and the signal is completely obscured. If it is still forced to predict the noise component, the network finds a low-loss shortcut: output the input itself. Mathematically, it degenerates into an identity mapping.

Diffusion fundamentally uses network outputs to estimate the data-distribution score $\nabla_z\log p_T(z)$. If the network only outputs its input at the first, high-noise sampling step, it provides a poor score estimate. DDPM can correct this gradually over hundreds or thousands of small steps, but a world model typically can afford only a few, such as fewer than 10 or even 1, to maintain agent-training efficiency. The first-step error then cannot be adequately corrected, substantially degrading generation quality.

### (2) Core algorithm

For a diffusion-based generative world model, suppose the initial noise at $t+1$ is $x_{t+1}$ and the previous context is $\psi_t$. An adaptive objective with a skip connection can be used:

$$
D_\theta(x_{t+1}^{\tau},\psi_t)
=c_{\mathrm{skip}}^\tau x_{t+1}^{\tau}
+c_{\mathrm{out}}^\tau F_\theta(c_{\mathrm{in}}^\tau x_{t+1}^{\tau},\psi_t)
$$

The actual training objective of network $F_\theta$ then becomes:

$$
\mathcal{L}(\theta)=
\mathbb{E}\left[
\left\|
F_\theta(c_{\mathrm{in}}^\tau x_{t+1}^{\tau},\psi_t) -
\frac{1}{c_{\mathrm{out}}^\tau}
\left(x_{t+1}^{0}-c_{\mathrm{skip}}^\tau x_{t+1}^{\tau}\right)
\right\|_2^2
\right]
$$

This provides a gradually changing target:

1. When noise is very small ($\tau\to 0$), $c_{\mathrm{skip}}\to 1$. The target becomes $x_{t+1}^0-x_{t+1}^{\tau}$, asking the network to predict the tiny residual noise.
2. Advantage: if the network still predicted the clean image $x^0$, input $x^\tau$ would already be almost identical to $x^0$. The output–input difference would be tiny, causing vanishing gradients and a trivial objective. Predicting residual noise amplifies these small differences, forcing fine refinement and high-frequency-detail improvement at the final stage.

### (3) Diffusion MoE

Dual- or multi-expert designs can improve adaptation. High-noise experts handle early denoising and global structure; low-noise experts handle later details.

## References

- Hafner, D., Yan, W., & Lillicrap, T. (2025). [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527). arXiv:2509.24527.
- Frans, K., Hafner, D., Levine, S., & Abbeel, P. (2025). [One Step Diffusion via Shortcut Models](https://arxiv.org/abs/2410.12557). ICLR.
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364). NeurIPS.
