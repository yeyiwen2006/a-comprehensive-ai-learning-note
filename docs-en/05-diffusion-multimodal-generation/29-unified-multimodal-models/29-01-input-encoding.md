---
title: "29.1 Input Encoding for Unified Multimodal Understanding and Generation Models"
chapter_title: "Unified Multimodal Understanding and Generation Models"
section_id: "29-01"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/29.统一多模态理解-生成模型/29.1 多模态统一理解-生成模型的输入编码.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 29.1 Input Encoding for Unified Multimodal Understanding and Generation Models

## I. Main Encoding Approaches

Directly discretizing continuous modalities such as images, as in LLMs, creates problems: images are information-dense, so direct discretization requires an enormous codebook; global semantic information for understanding and local information for generation readily conflict and are difficult to balance. Several main input-encoding approaches have therefore emerged.

### (1) Discrete encoding with shared understanding/generation encoding (BAAI Emu, LongCat-Next, ByteDance UniTok)

This approach generally converts images into discrete token IDs to fit common autoregressive architectures.

To process images like text, the system includes a visual tokenizer, usually based on VQ-VAE. It compresses and quantizes continuous image pixels into a discrete “visual vocabulary.” Images become sequences of visual tokens that have the same status as text tokens in format and dimensionality.

The LongCat-Next paper states that, with more training data, discrete representations can approach continuous-representation performance arbitrarily closely. Discrete representations have two advantages: they unify image and other modalities with LLM next-token prediction, and they are naturally compatible with RL methods such as GRPO, without requiring the ODE-to-SDE conversion needed for RL in flow-matching models.

The central difficulty is the limited capacity of a single finite discrete-token vocabulary. Possible solutions include:

1. Patchwise, hierarchical codebooks (LongCat-Next): rather than one “large dictionary,” the model uses 8*K “hierarchical dictionaries” (K is the number of image patches) and combines their results. This resembles how just 14 binary digits, each 0 or 1, represent 16384 combinations.

An image is first divided into patches. After VQ-VAE processing, each patch is tokenized through residual vector quantization (note that residual vector quantization itself has no parameters: it is a hard-coded selection of the best vocabulary match, while the vocabulary's embedding values are trained with the main model).

- **Level 1 ($l=1$)**: the first codebook matches the original image features, capturing the most central, abstract semantics (such as “this is a cat's ear”).
- **Compute the residual**: the first match is necessarily imperfect, leaving an error (residual = original features − first-level features).
- **Level 2 ($l=2$)**: the second codebook specifically quantizes this residual, capturing the next level of detail.
- **Repeat**: continue through level 8 ($l=8$). Each level repairs small errors from the previous one, capturing increasingly fine high-frequency textures and edges.

These 8 levels have their own vocabulary spaces and record global-to-local features. They are fused by addition: global features mainly support understanding, while local features mainly support generation, cleverly resolving the conflict.

Notably, ablations show worse performance if summed multilevel embeddings are not re-encoded. A single-layer FFN therefore remaps the summed features after codebook lookup to prevent semantic collapse from superimposed multilevel information.

2. Binary codebook (UniWeTok): in conventional VQ-VAE or VQGAN, the codebook is an explicit learnable parameter matrix. For each encoder output vector, distances to all codebook vectors are calculated and the minimum selected. Rich image information requires a huge codebook, making distance computation expensive. Binarization instead constructs a codebook from the sign of each vector dimension: positive becomes 1 and negative becomes −1, producing a codebook of size $2^{d_{\mathrm{model}}}$.

### (2) Discrete encoding with separate understanding/generation encoding (DeepSeek Janus)

Gradient conflicts between understanding and generation in one encoder likely arise because the tasks require different kinds of information. An intuitive solution is to encode them separately: understanding-encoder outputs mainly support text generation, while generation-encoder outputs mainly support visual generation.

Purely autoregressive Janus uses CLIP/SigLIP for understanding and a VQ tokenizer for generation. Only the understanding encoder is activated for understanding tasks, and vice versa.

### (3) Continuous encoding with separate understanding/generation encoding (ByteDance BAGEL, Meta Transfusion)

- **Text input**: converted directly into text tokens ($X_{\mathrm{text}}$) through the language model's own embedding layer (vocabulary lookup).
- **Image input**: the raw image (pixel matrix) is duplicated and fed into two visual encoders:
  - ViT outputs semantic tokens ($X_{\mathrm{ViT}}$) representing high-level concepts.
  - The VAE encoder outputs latent tokens ($X_{\mathrm{VAE}}$) representing continuous spatial structure.

Unlike Janus, which activates only the understanding encoder for understanding and vice versa, BAGEL concatenates all three token types and feeds them into a unified self-attention layer to better unify the tasks. When image generation begins, a `[BOI]` token is appended after the self-attention layer, followed by a number of noise tokens (such as 1024). These come from a noised target image during training and directly sampled white noise at test time. Parallel denoising and flow-velocity prediction then begin.

If VAE output dimensions still exceed the Transformer's input-vector dimension, linear-pooling downsampling or U-Net downsampling can be used. The latter requires more parameters but yields better generation quality.

Experiments in the Janus and BAEGL papers show that, with a shared encoder, adding generation significantly reduces understanding metrics and raises CE loss. Generation occupies semantic capacity needed for understanding, creating strong conflict. In the decoupled architecture (dual experts and dual encoders), generation no longer harms understanding, while understanding features provide semantic constraints for generation, improving text alignment and detail accuracy and enabling bidirectional cooperation.

### (4) Continuous encoding with shared understanding/generation encoding (RAE)

Earlier unified multimodal models effectively used two different feature languages for “seeing” and “drawing”:

- **Visual understanding (seeing)**: LLMs usually use representation encoders such as CLIP or SigLIP to understand images. Their features are high-dimensional and semantically rich (for example, 1152 dimensions), like the LLM's “visual native language.”
- **Visual generation (drawing)**: conventional diffusion models save compute by using a VAE to compress pixels into low-dimensional latents (for example, a few dozen dimensions), then denoise in that space.

- **The difficulty of separation**: diffusion produces low-dimensional VAE features, but the LLM recognizes only high-dimensional CLIP/SigLIP semantics. If diffusion draws a draft in VAE latent space, the LLM cannot understand it at all.
- **Conventional workflow**: to check a diffusion-generated image, the LLM must follow a cumbersome chain: diffusion generates VAE features -> the VAE decoder restores pixels -> the CLIP encoder re-encodes pixels into high-dimensional features -> the LLM evaluates them. The model thus resembles two brains unable to communicate.

RAE's core idea is direct: if the LLM is accustomed to SigLIP's high-dimensional semantics, why not force diffusion to generate those features directly?

RAE therefore does not remove the encoder; it discards the VAE encoder entirely and uses SigLIP-2 as the sole shared encoder.

- **Unified generation space**: diffusion no longer predicts low-dimensional VAE features, but denoises directly in the $16\times16\times1152$-dimensional SigLIP-2 semantic space.
- **Seamless modality communication**: diffusion directly generates SigLIP-2 features, exactly the LLM's “native language.” As soon as diffusion forms a draft in latent space, the LLM reads the feature sequence and judges whether the image meets requirements, without pixel decoding and re-encoding.

Earlier unified models also often used low-dimensional compressed representations to accommodate generation compute, failing to provide both the semantic capacity for understanding and the fine detail for generation. This is a major source of task conflict.

Unlike VAE encoders and decoders with equal parameter counts, RAE uses a heavy encoder and light decoder:

- **Heavy encoder**: directly uses an open-source visual foundation model with a very large parameter count (such as billion-parameter SigLIP-2, DINOv2 or MAE). It carefully “sees through” real images to extract very high-dimensional, semantically rich representations.
- **Light decoder**: a shallow ViT with few layers and very few parameters. Because the heavy encoder's features already contain complete information, it needs no complex semantic reasoning. It merely “translates” the high-dimensional feature matrix into a visible RGB pixel matrix.

Diffusion in high-dimensional space is expensive, so the light decoder reduces compute and balances the cost. (The encoder is not needed when generating images.)

In summary, RAE does not return to low-level pixel space; it raises generation into the same high-dimensional semantic space as understanding.

With scale, mechanisms such as a wider diffusion head and noise-augmented decoding offer almost no further gains. Some mechanisms must remain, however:

- **Essential core mechanism: dimension-aware noise scheduling**. High- and low-dimensional spaces differ, so time steps must be shifted and scaled according to effective data dimension $m$ (token count times feature dimension). Removing this causes catastrophic performance degradation. The time shift is:

$$
t_m=\frac{\alpha t_n}{1+(\alpha-1)t_n}
$$

Here, $n$ is the baseline dimension and $\alpha=\frac{m}{n}$.

RAE also avoids overfitting more effectively than VAE:

In modern T2I training with “large-scale pretraining plus high-quality small-data fine-tuning,” overfitting is a critical problem.

- **VAE fragility**: experiments observe catastrophic overfitting after only 64 fine-tuning epochs on high-quality data. Training loss rapidly plummets, and the model memorizes samples and loses generalization.
- **RAE stability**: RAE instead shows remarkable stability, maintaining performance even after 256 or 512 epochs. The paper hypothesizes that its high-dimensional, highly semantic latent space provides implicit regularization, forcing learning of real feature distributions rather than memorization of pixel details.

Subsequent work, Beyond Language Modeling, shows that high-dimensional RAE representations improve both generation and understanding. It argues that the conventional “LLM-dominant, vision-subordinate” grafted architecture prevents cooperation by forcing visual representations to align with text, sacrificing fine-grained understanding or generation precision and exacerbating modality conflict. Its main experiments examine mutual gains from generation and understanding, using a unified Transformer with RAE representations pretrained from scratch on mixtures of text-only, video-only, image–text pairs and action-video data. Experiments and conclusions:

1. Bidirectional gains: with a fixed visual-token budget, adding text tokens continually improves generation; with a fixed text-token budget, adding visual tokens does not harm language modeling, demonstrating no negative conflict between vision and language.
2. Generalization gains: 20B of VQA data plus 80B of general multimodal data (video/image–text/text) outperforms training on 100B of VQA-only data, showing that generation-related multimodal data benefits visual understanding.
3. Paradigm comparison: native joint multimodal pretraining significantly outperforms grafting vision onto an LLM, leading across both tasks and showing that equal-status modality architectures are key to mutual gains.

### (5) Direct pixel patches without an encoder, with shared understanding/generation (SenseTime SenseNova-U1)

Many unified multimodal understanding/generation models remain divided in practice. First, the understanding side learns semantics with a visual encoder, while generation learns reconstructable features through VAE latents. These are not naturally aligned; different modalities rely on different tokenizers, diffusion heads or auxiliary modules, often pretrained separately and then combined for fine-tuning. The result is frequently unified only superficially. Second, discrete visual tokens lose detail, while VAE latents impose a compression bottleneck: the former harms fidelity, the latter semantic expression and end-to-end learning.

SenseNova-U1 proposes neither converting images into pretrained visual-encoder features nor compressing them into VAE latents. Instead, it directly processes pixel patches and text tokens, learning both understanding and generation in one architecture. It uses neither a conventional pretrained vision encoder nor a VAE, but lightweight patch encoding/decoding layers as its visual interface. At image input:

Image / noisy image -> two convolutional layers + GELU + 2D positional encoding -> visual patch tokens

Convolution strides are 16 and 2, so each visual token corresponds to a $32\times32$ patch. The image region is enclosed by `<img>` and `</img>` markers, while text uses the underlying language model's original tokenizer. Image and text tokens enter the same embedding space and are then processed jointly by the Transformer.

Visual-generation diffusion uses resolution-adaptive noise. Different resolutions have different patch counts: an $N\times N$-pixel image has $N\times N/1024$ tokens, and noise addition and removal operate on each token. Mathematically, with $N$ approximately independent noise tokens, the total random fluctuation scale grows roughly with $\sqrt{N}$. Larger images therefore require noise scale to grow with the square root of the token count; otherwise, high-resolution images begin with relatively “narrow” noise and are “insufficiently disrupted,” putting their flow trajectories on a different scale from low-resolution samples.

## II. Embedding Training Methods

The losses below are often combined.

### (1) Contrastive learning

Mainstream visual encoders such as SigLIP and CLIP use this approach to bring global image and text features closer.

$$
\mathcal{L}_{\mathrm{contrastive}}
=-\log\frac{\exp(s(I,T)/\tau)}
{\sum_j\exp(s(I,T_j)/\tau)}
$$

This aligns high-level semantics but loses details, making it unsuitable for generation.

### (2) Reconstruction loss

To preserve details, a VAE-like loss can combine MSE, perceptual loss (similarity computed after extracting features with a pretrained neural network), and GAN loss. ROSS, from the Chinese Academy of Sciences, MEGVII and ModelBest, experimentally shows that reconstruction loss improves understanding, whereas text-to-image training does not necessarily do so, because the former provides fine-grained, stable gradients.

### (3) Native multimodal pretraining

If the main model is pretrained from scratch, including global and fine-grained visual understanding and generation tasks lets the encoder learn representations that preserve both high-level semantics and low-level details.

For images of different resolutions, adaptive-resolution methods can be used, or the model can be pretrained at fixed resolution and adapted to multiple resolutions during fine-tuning. Face data and text-containing images can receive specialized fine-tuning.

## III. Special Training Methods for Discrete Encoding

### (1) Training discrete encoders with continuous encoders

A discrete encoder differs from a continuous one only by an additional nearest-neighbor codebook lookup at the end. A continuous encoder can therefore be used to train a semantically complete continuous encoder. Specific methods include:

1. Distill a continuous encoder: train the discrete encoder to imitate its outputs, using cosine-similarity loss to align activations at every layer with the corresponding continuous-encoder layer.

For discrete codebooks, UniWeTok proposes distillation both before and after quantization. Post-quantization distillation prevents quantization from losing distilled information. Pre-quantization distillation is needed because, as discussed later, the loss gradient for a discrete codebook vector is forcibly replaced with the gradient for the encoder's continuous vector, although these gradients are not actually equal. Post-quantization distillation alone therefore gives inaccurate backpropagated gradients.

2. Fine-tune using a continuous encoder's weights as initialization.

### (2) Additional losses for discretization

1. Commitment loss

$$
\alpha\lVert U_G-\mathrm{sg}[Q]\rVert^2
$$

- **Meaning**: $U_G$ is the encoder's continuous latent output, and $Q$ is the quantized feature from the codebook. $\mathrm{sg}[\cdot]$ denotes stop-gradient.
- **Purpose**: because quantization loses information, this loss forces continuous features $U_G$ to “commit” to or approach selected discrete features $Q$, preventing arbitrary wandering in latent space and stabilizing quantization.

2. Entropy loss

This has two parts. Token entropy loss asks the model to be decisive for each individual token, minimizing uncertainty. Codebook entropy loss asks it to spread usage across the overall data, with appropriately high uncertainty to use the full codebook rather than only a few tokens.

3. Generative-Aware Prior (GAP)

A lightweight “proxy generator” can provide next-token supervision for corresponding image generation during tokenizer training, forcing token distributions themselves to be friendly to autoregressive generation.

### (3) Codebook training and STE

Codebook vectors are initialized randomly or through clustering. Training often encounters the following problem:

In a VQ architecture, the encoder outputs continuous vector $z_e$, the quantizer looks up discrete codebook vector $z_q$ (namely $e_k$), and the decoder reconstructs an image from $z_q$.

Nearest-neighbor quantization is:

$$
k=\arg\min_j\lVert z_e-e_j\rVert_2
$$

$$
z_q=e_k
$$

The gradient problem comes from the mathematical nature of $\arg\min$, a piecewise step function. A small change in encoder output $z_e$ has two possibilities:

1. Usually, the current $e_k$ remains closest, so selected index $k$ and $z_q$ do not change. The derivative is 0.
2. Rarely, $z_e$ crosses a boundary and index $k$ changes abruptly. The derivative is infinite ($\infty$).

Training therefore uses a straight-through estimator (STE). For continuous encoder output $z_e$, the discrete encoder's final forward output $z_q$ is assigned codebook value $e_k$. In backpropagation, the loss gradient for final output $z_q$ is directly replaced by that for continuous output $z_e$:

$$
z_q=z_e+\mathrm{sg}[e_k-z_e]
$$

The decoder's gradient $\frac{\partial L}{\partial z_q}$ does not become 0; it passes through the quantization layer unchanged, in a 1:1 ratio, directly to the encoder.

However, this makes the gradient of $L$ with respect to codebook vector $e_k$ equal to 0, requiring other mechanisms to update the codebook. Two common methods are:

**Mechanism A: Codebook loss — the classical gradient-based approach**

An additional penalty is explicitly added to the total loss.

For the selected codebook vector $e_k$ and corresponding continuous encoder feature $z_e$ in each forward pass, compute MSE:

$$
\mathcal{L}_{\mathrm{codebook}}
=\lVert \mathrm{sg}[z_e]-e_k\rVert_2^2
$$

- **Principle**: $\mathrm{sg}[\cdot]$ is stop-gradient, treating encoder output $z_e$ as a fixed target anchor within this loss.
- **Update**: the optimizer updates only $e_k$, moving it step by step toward encoder output $z_e$ in continuous space.

Note: to keep the encoder from wandering, a corresponding commitment loss $\lVert z_e-\mathrm{sg}[e_k]\rVert_2^2$ also pulls its output toward the codebook, making both sides approach each other.

**Mechanism B: Exponential moving average (EMA) — a more stable modern approach**

In large-scale VQ-VAE or VQ-GAN training, including the underlying design of UniTok, gradient-descent codebook updates are often unstable. Modern architectures therefore tend to abandon optimizer-based updates and use EMA.

- **Update procedure**:

1. In each batch, record the continuous features $z_e$ assigned to codebook vector $e_k$.
2. Compute the mean of all $z_e$ assigned to $e_k$ in this batch, denoted by $\bar z$.
3. Update $e_k$ directly using:

$$
e_k^{(\mathrm{new})}
=\gamma\cdot e_k^{(\mathrm{old})}+(1-\gamma)\cdot\bar z
$$

Here, $\gamma$ is a decay coefficient such as 0.99.

- **Advantage**: EMA requires no gradients. It acts like dynamic online K-means, smoothly and stably tracking changes in the encoder-feature distribution.

**Mechanism C: UniTok's special update (semantic-distillation injection)**

Besides the basic reconstruction update that pulls codebooks toward image features, UniTok's codebook also performs semantic alignment.

In contrastive objectives such as image–text contrastive loss, the codebook vector $e_k$ for an activated discrete token is compared with the corresponding continuous text vector through cosine similarity. Additional gradients force $e_k$ not only to resemble the original patch but also to move into alignment with its text description in high-dimensional continuous space.

To ensure full codebook utilization, rarely used tokens are also replaced:

The system periodically scans the codebook. If a vector $e_k$ is used below a threshold, it is discarded and overwritten with an active vector randomly drawn from current-batch encoder features $z_e$. This “revives” the dead $e_k$ for subsequent EMA or gradient updates.

### (4) The SigLu technique for binary-codebook training: UniWeTok

Binary-codebook quantization creates a conflict: token entropy loss pushes encoder outputs toward $\pm\infty$, while commitment loss pulls them toward codebook entries $\pm1$. Applying $\mathrm{SigLu}(x)=\frac{1-e^x}{1+e^x}$ constrains encoder outputs to $(-1,1)$. Under this constraint, the two losses are mathematically equivalent.

## References

- Meituan LongCat Team. (2026). [LongCat-Next: Lexicalizing Modalities as Discrete Tokens](https://arxiv.org/abs/2603.27538). arXiv:2603.27538.
- Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., & Luo, P. (2024). [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848). arXiv:2410.13848.
- Ma, C., Jiang, Y., Wu, J., Yang, J., Yu, X., Yuan, Z., Peng, B., & Qi, X. (2025). [UniTok: A Unified Tokenizer for Visual Generation and Understanding](https://arxiv.org/abs/2502.20321). arXiv:2502.20321.
- Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., Shi, G., & Fan, H. (2025). [Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683). arXiv:2505.14683.
- Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., & Levy, O. (2024). [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039). arXiv:2408.11039.
- Zhuang, S., Ai, Y., Han, J., Mao, W., Li, X., Wang, F., Wang, X., Li, Y., Lin, S., Xu, K., Yang, Z., Huang, H., Yue, X., Chen, H., & Wang, Y. (2026). [UniWeTok: An Unified Binary Tokenizer with Codebook Size $2^{128}$ for Unified Multimodal Large Language Model](https://arxiv.org/abs/2602.14178). arXiv:2602.14178.
- Zheng, B., Ma, N., Tong, S., & Xie, S. (2025). [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690). arXiv:2510.11690.
- Tong, S., Fan, D., Nguyen, J., Brown, E., Zhou, G., Qian, S., Zheng, B., Vallaeys, T., Han, J., Fergus, R., Murray, N., Ghazvininejad, M., Lewis, M., Ballas, N., Bar, A., Rabbat, M., Verbeek, J., Zettlemoyer, L., Sinha, K., LeCun, Y., & Xie, S. (2026). [Beyond Language Modeling: An Exploration of Multimodal Pretraining](https://arxiv.org/abs/2603.03276). arXiv:2603.03276.
- Diao, H., Wu, P., Deng, H., et al. (2026). [SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture](https://arxiv.org/abs/2605.12500). arXiv:2605.12500.
- Wang, H., Zheng, A., Zhao, Y., Wang, T., Ge, Z., Zhang, X., & Zhang, Z. (2024). [Reconstructive Visual Instruction Tuning](https://arxiv.org/abs/2410.09575). arXiv:2410.09575.
