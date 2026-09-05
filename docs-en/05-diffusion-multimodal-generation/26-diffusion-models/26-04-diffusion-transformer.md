---
title: "26.4 Diffusion Transformer"
chapter_title: "Diffusion Models"
section_id: "26-04"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/26.扩散模型/26.4 Diffusion Transformer.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 26.4 Diffusion Transformer

## I. Core Architecture and Workflow

Diffusion Transformer introduces the Transformer architecture into diffusion models, unifying their architecture with that of large language models and improving scalability, so that generation quality can improve as the number of model parameters increases.

1. **Latent patchification**

DiT does not operate directly on pixels, but on latent features encoded by a variational autoencoder (VAE).

Given an input image $x\in\mathbb{R}^{H\times W\times 3}$, a pretrained VAE encoder first maps it into latent space, producing the latent representation $z\in\mathbb{R}^{h\times w\times c}$.

Next, DiT divides $z$ into spatial patches of size $p\times p$, flattens these patches, and applies a linear projection to convert them into a token sequence of length $T=(h/p)\times(w/p)$ and dimension $d$. Finally, standard learnable positional embeddings are added.

2. **Conditioning mechanism: the adaLN-Zero block**

Standard diffusion models need to inject the current time step $t$ and conditioning class $c$ at every denoising step. U-Net usually does this through cross-attention or feature concatenation, whereas DiT uses adaptive layer normalization (adaLN).

This is the most important design in the DiT architecture. For a hidden state $h$ entering a DiT block, adaLN operates as follows:

$$
\mathrm{adaLN}(h,t,c)=\gamma_{(t,c)}\mathrm{LayerNorm}(h)+\beta_{(t,c)}
$$

Here, the scale parameter $\gamma_{(t,c)}$ and shift parameter $\beta_{(t,c)}$ are generated directly by applying a multilayer perceptron (MLP) regressor to the sum of the time-step and class embeddings, rather than being fixed in the network as learnable parameters.

In addition, DiT introduces a **dimension-scaling gating mechanism** before the residual connections. To greatly stabilize training, the authors propose the **adaLN-Zero** initialization strategy: the MLP is forced to output zeros for $\gamma_{(t,c)}$, $\beta_{(t,c)}$, and the residual gating parameter $\alpha_{(t,c)}$ at initialization.

This means that, at the start of training, **every DiT block is a strict identity function**:

$$
h_{l+1}=h_l+0=h_l
$$

This ensures that the network does not collapse at initialization because of the high variance of a deep Transformer.

3. **Unpatchification and prediction**

After passing through $N$ DiT blocks, the token sequence is sent to a final standard fully connected layer, which expands the dimension back to $p\times p\times 2c$ (the factor of 2 is needed because the model predicts both noise and variance). The one-dimensional sequence is then reshaped into a spatial grid of $h\times w\times 2c$ and passed on for loss computation or sampling.
We find that adaLN can inject only a fixed-length vector $c$, such as a `[CLS]` token. Clearly, however, a single token is insufficient, so MM-DiT, based on multimodal dual-stream attention, can be introduced (models that combine diffusion and autoregression, with increasingly long autoregressive contexts, also use this method):

First, adaLN is applied separately to the image and text streams, and each stream computes its own Q, K, and V. The image QKV and text QKV are then concatenated directly, and global self-attention is performed on this long concatenated sequence. Image and text can each integrate information internally; images can also extract the corresponding text prompts, while text updates its context based on the images.

After attention is computed, the long output sequence is split again into image features and text features. These are then sent to their respective residual connections and feedforward networks (FFNs), with the gating coefficients predicted by adaLN applied.

(The image at this point repeats the descriptions of adaLN-Zero initialization and unpatchification above and has been incorporated into the transcription.)

4. Differences between the attention module and an autoregressive Transformer

In a diffusion model, all tokens in any attention layer must compute attention with one another simultaneously, and the vectors for all tokens are also output simultaneously. This requires an actual $N\times N$ attention-score matrix to be instantiated in GPU memory. Moreover, if diffusion takes $K$ steps, this matrix must be computed $K$ times. For video generation models, $N=\text{number of frames}\times\text{number of patches}$, which is why many video generation models use “diffusion within blocks and autoregression between blocks.”

**Training workflow**

1. **Image encoding**: feed the original image $x$ into the frozen VAE encoder to extract latent features $z_0$.
2. **Random noise addition**: randomly sample a time step $t\sim U(1,T)$ and standard Gaussian noise $\epsilon\sim\mathcal{N}(0,I)$. Compute the noisy latent variable $z_t$ using the forward formula.
3. **Patchification**: divide $z_t$ into patches, flatten them, and add positional embeddings to form the input sequence.
4. **Condition embedding**: convert time step $t$ and the image's class label $c$ into conditioning vectors (through embedding layers and MLPs).
5. **DiT forward pass**:
   - Feed the token sequence into $N$ DiT blocks.
   - Within each block, the conditioning vector dynamically generates adaLN scale and shift parameters to adaptively normalize the tokens.
6. **Unpatchification**: restore the final output sequence to the same spatial dimensions as $z_t$.
7. **Loss computation and backpropagation**: the network outputs the predicted noise $\epsilon_\theta$ and the diagonal elements of the covariance matrix. Compute $L_{\mathrm{simple}}+L_{\mathrm{vlb}}$ and update the Transformer weights through gradient descent.

**Inference (generation) workflow**

1. **Noise initialization**: sample a pure-noise tensor $z_T\sim\mathcal{N}(0,I)$ from a standard normal distribution, and specify the desired class label $c$.
2. **Iterative denoising loop**: for time steps $t=T,T-1,\ldots,1$, perform the following:
   - Divide the current $z_t$ into a sequence of patches.
   - Feed the sequence, together with time step $t$ and class $c$, into the DiT network.
   - The network processes the context through the adaLN-Zero mechanism and predicts the current step's noise $\epsilon_\theta$ and variance $\Sigma_\theta$.
   - Use these predictions and the mathematical formulas of the DDPM or DDIM sampling algorithm to compute the previous step's latent variable $z_{t-1}$.
3. **Image decoding**: the loop produces the fully denoised $z_0$. Feed $z_0$ into the frozen VAE decoder to reconstruct the final high-resolution pixel image $\hat{x}$.

## II. Training Objective

During DiT training, the network $\theta$ must predict both the added noise $\epsilon_\theta$ and the variance $\Sigma_\theta$.

Noise prediction uses the simplified mean squared error loss:

$$
L_{\mathrm{simple}}=\mathbb{E}_{z_0,\epsilon,t}\left[\lVert \epsilon-\epsilon_\theta(z_t,t,c)\rVert_2^2\right]
$$

To learn the variance $\Sigma_\theta$, the model additionally introduces the full variational lower bound (VLB) loss. The total loss combines the two, usually applying a small weight to $L_{\mathrm{vlb}}$ to avoid disrupting optimization of the main objective:

$$
L=L_{\mathrm{simple}}+L_{\mathrm{vlb}}
$$

## III. Variance Prediction

### (1) Why does the objective include a variance-prediction term?

In the reverse denoising process $p_\theta(z_{t-1}\mid z_t)$, we need to assume a Gaussian distribution:

$$
p_\theta(z_{t-1}\mid z_t)=\mathcal{N}(z_{t-1};\mu_\theta(z_t,t),\Sigma_\theta(z_t,t))
$$

In the original DDPM (Ho et al., 2020), the authors found two theoretical extreme bounds (an upper and a lower bound) for the true posterior variance:

1. **Upper bound**: $\beta_t$ (assuming the real image consists entirely of standard normal noise).
2. **Lower bound**: $\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ (assuming the real image $z_0$ is completely known and deterministic).
More specifically:

Since the model can only see a noisy image $z_t$, from its perspective $z_0$ is certainly not a definite point, but **a probability distribution containing countless possibilities**.

We can construct an intuitive thought experiment:

- **Extreme case A ($t$ is large, close to pure noise)**: suppose $z_t$ looks like static on a television with no signal. Looking at that static, can the model determine whether the original scene showed a cat, a dog, or a car? Not at all. Thousands of different $z_0$ values collapse to the same static after a large amount of noise is added. The model is then extremely uncertain about the true $z_0$, making its inferred variance of the previous image very large, approaching the theoretical upper bound $\beta_t$.
- **Extreme case B ($t$ is small, near the end of denoising)**: suppose $z_t$ is a very clear photograph of a cat, with only a few dust particles on it (small noise). Looking at this image, the model is highly confident that it originally depicted a cat (other possibilities for $z_0$ now have very low probability). The model is then very certain about $z_0$, and the reverse-denoising variance is very small, approaching the theoretical lower bound $\tilde{\beta}_t$.
During training, we know that the real image is in the training set, fully known and deterministic, but the model does not. During inference, if the model has no confidence at all about what the true $z_0$ is, the variance of $z_{t-1}$ takes the upper bound; if it is completely certain about the true $z_0$, the variance of $z_{t-1}$ takes the lower bound.

Why the upper bound is $\beta_t$:

In the reverse process, we want to derive $q(z_{t-1}\mid z_t)$. By Bayes' formula, we can expand it as:

$$
q(z_{t-1}\mid z_t)=\frac{q(z_t\mid z_{t-1})q(z_{t-1})}{q(z_t)}
$$

- $q(z_t\mid z_{t-1})$ is the forward noise-addition process and is fully known: $\mathcal{N}(z_t;\sqrt{\alpha_t}z_{t-1},\beta_t I)$.
- **The key point**: in the extreme case of maximum uncertainty (usually when $t$ is large, near the end of diffusion), the image has been corrupted beyond recognition. If we assume we have no knowledge of $z_0$, the most reasonable assumption for the marginal distribution of $z_{t-1}$ is that it has already degenerated into a **standard normal distribution (the prior)**:

$$
q(z_{t-1})\approx\mathcal{N}(0,I)
$$

We then have:

$$
q(z_{t-1}\mid z_t)\propto q(z_t\mid z_{t-1})q(z_{t-1})
$$

$$
q(z_{t-1}\mid z_t)\propto \exp\left(-\frac{1}{2}\left[\frac{(z_t-\sqrt{\alpha_t}z_{t-1})^2}{\beta_t}+z_{t-1}^2\right]\right)
$$

After completing the square, the variance can be determined from the coefficients of the normal distribution.

The original DDPM argued that, as long as the total number of diffusion steps $T$ is sufficiently large (for example, $T=1000$), the change at each step is tiny, so $\beta_t$ and $\tilde{\beta}_t$ are almost equal. Therefore, **the original model chose not to predict variance**, instead fixing it directly to a constant matrix:

$$
\Sigma_\theta(z_t,t)=\sigma_t^2 I
$$

Here, $\sigma_t^2$ is hard-coded as $\beta_t$ or $\tilde{\beta}_t$.

**Critical limitation:**

This works well for $T=1000$, but if we want to **accelerate sampling**, for example by using only 50 steps (large-step sampling), the two variance bounds diverge substantially. If a fixed variance is still used, the model is forced into a suboptimal probability distribution, producing images that are noisy or extremely blurry.

### (2) How is variance predicted?

To address variance uncertainty in large-step sampling, Improved DDPM proposed letting the neural network dynamically predict the most appropriate variance from the current features $z_t$ and time step $t$, a method also adopted by DiT.

1. **Variance parameterization**

Having the model output an absolute variance value directly is unstable. Instead, it is designed to output an interpolation-coefficient vector $v\in[0,1]^d$, which linearly interpolates between the theoretical upper and lower bounds in log space:

$$
\log\Sigma_\theta(z_t,t)=v\odot\log\beta_t+(1-v)\odot\log\tilde{\beta}_t
$$

Note: this is also why DiT's final fully connected layer expands the feature dimension to $2c$. Of these, $c$ channels predict the noise $\epsilon_\theta$, while the other $c$ channels predict this interpolation vector $v$.

2. **Stop-gradient and the loss function**

To train this dynamic variance, we must introduce the full variational lower bound (VLB) loss:

$$
L_{\mathrm{vlb}}=\sum_{t=1}^{T}D_{\mathrm{KL}}\left(q(z_{t-1}\mid z_t,z_0)\parallel p_\theta(z_{t-1}\mid z_t)\right)
$$

Because computing $L_{\mathrm{vlb}}$ involves both the mean $\mu_\theta$ and the variance $\Sigma_\theta$, optimizing it directly makes the mean-prediction gradients extremely unstable and harms image quality.

DiT therefore applies **stop-gradient** to the mean prediction when computing $L_{\mathrm{vlb}}$. This means that the backpropagated gradients from $L_{\mathrm{vlb}}$ only update the weights for variance $v$, without interfering with the learning of noise $\epsilon_\theta$.

The final hybrid loss is:

$$
L=L_{\mathrm{simple}}+\lambda L_{\mathrm{vlb}}
$$

In DiT, $\lambda=0.001$ is usually used to ensure that variance learning does not interfere with the main noise-prediction task.

## References

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR.
- Peebles, W., & Xie, S. (2023). [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). ICCV.
- Esser, P., Kulal, S., Blattmann, A., et al. (2024). [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206). arXiv:2403.03206.
- Nichol, A. Q., & Dhariwal, P. (2021). [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). ICML.
