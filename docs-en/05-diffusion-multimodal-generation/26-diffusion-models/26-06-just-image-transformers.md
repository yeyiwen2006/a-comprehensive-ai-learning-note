---
title: "26.6 Just Image Transformers"
chapter_title: "Diffusion Models"
section_id: "26-06"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/26.扩散模型/26.6 Just Image Transformers.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 26.6 Just Image Transformers

## I. Core Idea

Today's mainstream diffusion models (such as DDPM) and flow-matching models usually train neural networks to predict noise $\epsilon$ or velocity $v$. JiT points out that predicting clean data (x-prediction) and predicting noisy quantities ($\epsilon$- or v-prediction) are fundamentally different mathematically.

This conclusion rests on the **manifold assumption**:

- **Clean images**: denoted by $x$, natural images are not distributed uniformly throughout the high-dimensional pixel space $\mathbb{R}^D$, but are highly concentrated on a very low-dimensional manifold $\mathcal{M}$ ($d\ll D$).
- **Noise**: denoted by $\epsilon$, Gaussian noise is isotropic, lies off the manifold, and is spread throughout the high-dimensional space $\mathbb{R}^D$.
- **Velocity**: denoted by $v$, in flow matching $v=\epsilon-x$, which is likewise an off-manifold quantity in high-dimensional space.

When a network is asked to predict $\epsilon$ or $v$, it is forced to fit an unstructured mapping spanning the entire high-dimensional space. When it is asked to predict $x$ directly, its prediction target is strictly restricted to the low-dimensional natural-image manifold $\mathcal{M}$. This reduction in target dimensionality greatly reduces the difficulty of fitting the neural network, allowing the model to operate efficiently in high-resolution pixel space (such as $512\times512$) without relying on dimensionality reduction in latent space.

## II. Model Architecture

JiT's architecture can be described as “nothing but a standard image Transformer.”

1. **No tokenizer and no latent space**: it completely discards the pretrained VAE used by mainstream architectures and operates directly on raw pixels.
2. **Large patch sizes**: to address the explosion in sequence length caused by high-resolution images, JiT uses very large patch sizes (such as $16\times16$, or even $32\times32$ and $64\times64$).
3. **Bottleneck design**: experiments show that, in x-prediction mode, substantially compressing the Transformer's linear embedding dimension does not cause collapse and instead preserves robustness. This indirectly supports the manifold assumption: because the target $x$ is intrinsically low-dimensional, a low-capacity network bottleneck is sufficient to capture its features; reducing network capacity when predicting high-dimensional noise $\epsilon$, by contrast, causes catastrophic failure.
4. **No pretraining and no additional losses**: no form of pretraining is required, and neither perceptual loss nor adversarial loss is needed.

The only additional module is a fully connected layer after the Transformer, which maps the generated low-dimensional embedding vectors into a high-dimensional image. This works because high-dimensional image patches from the real world are not random and disordered: they exhibit strong spatial coherence and regularity and are therefore tightly constrained to a very low-dimensional “manifold.” The low-dimensional hidden variables inside the Transformer are already sufficient to capture this low-dimensional structure. The final fully connected layer essentially just “rotates” and “maps” this structure back to the corresponding manifold location in high-dimensional space.

As in Diffusion Transformer, the time step is injected through adaptive layer normalization.

1. **Time embedding**:

   First, the scalar time step $t$ is mapped to a high-dimensional vector through sinusoidal positional encoding. An MLP then extracts features from it, producing a global time-feature vector $E_t$.

2. **Generate modulation parameters**:

   A simple linear regression layer is placed outside each Transformer block. It takes $E_t$ as input and directly regresses the sets of modulation parameters needed by that block (usually a scale factor $\gamma$, a shift factor $\beta$, and a gating factor $\alpha$ for the residual connection).

3. **Adaptive modulation**:

   Before the input visual-token sequence enters the multi-head self-attention (MSA) layer or feedforward network (FFN), standard layer normalization is applied to the tokens. The regressed $\gamma$ and $\beta$ then apply an elementwise affine transformation to the normalized features:

$$
\mathrm{AdaLN}(x,t)=\gamma(t)\cdot\mathrm{LayerNorm}(x)+\beta(t)
$$

Here, the normalization scale and offset are determined dynamically and entirely by the current time step $t$.

4. **Zero initialization**:

   This is the central innovation of adaLN-Zero. The weights and biases of the linear layer that regresses $\gamma,\beta,\alpha$ are set strictly to $0$ at network initialization. At the start of training, the scale factor therefore has no effect, the shift is zero, and the residual-block gating factor is $\alpha=0$. This makes every Transformer block equivalent to an identity mapping at initialization. This design greatly stabilizes gradient flow in the early training of deep networks, which is especially important when training large Transformers directly in raw high-dimensional pixel space.

## III. Why Does Industry Still Use Conventional Diffusion Models?

1. Latent space already avoids the curse of dimensionality

Kaiming He's paper primarily addresses the curse of dimensionality in **raw pixel space**, whereas almost all current industrial world models use latent diffusion architectures.

**Latent space already avoids the “curse of dimensionality.”** These models use a variational autoencoder (VAE) to compress extremely high-dimensional visual/image frames into a very low-dimensional latent space $z\in\mathbb{R}^k$.

Because a VAE is itself a highly capable manifold learner, latent space is already a heavily compressed low-dimensional manifold. In such a relatively low-dimensional space, the difficulty and dimensionality penalty of predicting noise $\epsilon$ or velocity $v$ are greatly reduced.

If diffusion is performed directly on raw pixels, even a $1$-minute $1080$P video has a dimensionality as high as:

$$
1920\times1080\times3\times60\times60\times2.2\times10^{10}
$$

Computing gradients and performing Markov-chain sampling in such a remarkably high-dimensional space is not only prohibitively expensive but also encounters a severe “curse of dimensionality.”

2. Diversity of the action space

To solve these problems, embodied AI introduced Diffusion Policy. The diffusion model completely abandons the direct output of action values and instead learns the energy field or score function of the action distribution.

Workflow and the Langevin dynamics mechanism:

1. **Learn the score function (score matching)**: the core training task of the diffusion model is to train a network to fit the gradient of the real data distribution with respect to the data itself; the score is $\nabla_a\log p(x)$. It can also be imagined as a compass pointing toward the “most reasonable action.”
2. **Reverse denoising sampling (Langevin dynamics)**: during inference, the model starts with pure Gaussian random noise $x_T\sim\mathcal{N}(0,I)$ and updates the state step by step as follows:

$$
x_{t-\Delta t}=x_t+\frac{1}{2}\Delta t\nabla_x\log p(x_t)+\sqrt{\Delta t}z
$$

Here, $z\sim\mathcal{N}(0,I)$ is the injected random noise.

Why does it not fall into the valley (gap)?

- **Gradient ascent (seeking a mode)**: $\nabla_x\log p(x_t)$ in the formula pulls the sample toward regions of high probability, where the actions are optimal, moving it from low-density regions toward high-density ones.
- **Noise injection (breaking symmetry)**: $\sqrt{\Delta t}z$ at the end of the formula is the source of noise. The “mean point” (gap point) between the left and right peaks is actually a probabilistic saddle point or local valley. The randomly injected noise breaks the balance and pushes the generated trajectory randomly toward the left or right “basin of attraction,” eventually placing it in a definite, safe mode (action $A$ or action $B$), thereby achieving iterative refinement and multimodal decision-making perfectly.

Note: p(x) here is not the joint probability density of the training data, but the probability distribution of the new data itself.

3. Stability under high-noise conditions

At the beginning of diffusion (as $t\to T$), the image is almost pure Gaussian noise. Information about $x_0$ is then heavily obscured in $x_t$. Mathematically, asking the model to predict an extremely clear $x_0$ directly from pure noise causes severe numerical instability because of excessive variance, and the model often outputs an extremely blurry average image. Predicting noise $\epsilon$ or velocity $v$ at this stage has been shown to be more stable in terms of numerical gradients and loss weighting (SNR weighting), which is also why flow matching is so widely used in industry.

$$
v=x_1-x_0
$$

Here, $x_1$ is pure noise and $x_0$ is a clear image.

Algebraic manipulation reveals that predicting $v$ is effectively a dynamic, automatic interpolation between “predicting noise” and “predicting the original image,” according to the signal-to-noise ratio at the current time step.

It avoids both the variance explosion of early $x_0$ prediction and the signal-to-noise problem of late $\epsilon$ prediction. Moreover, because the trajectory is direct (an ODE), it greatly reduces the number of inference steps. These are the properties that world models (generating high-frame-rate video) truly need.

## References

- Li, T., & He, K. (2025). [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720). arXiv:2511.13720.
- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS.
- Chi, C., Xu, Z., Feng, S., et al. (2023). [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137). Robotics: Science and Systems.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR.
