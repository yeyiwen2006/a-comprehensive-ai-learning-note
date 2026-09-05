---
title: "28.7 Context Engineering for Video Generation Models"
chapter_title: "Multimodal Generation"
section_id: "28-07"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/28.多模态生成/28.7 视频生成模型的上下文工程.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 28.7 Context Engineering for Video Generation Models

## I. Spatiotemporal Context Compression

### (1) Background

**Problem to solve**: during autoregressive long-video generation, increasing historical frames $z_c$ makes the Transformer's (DiT's) context length grow linearly or even quadratically, exploding GPU memory usage and sharply reducing inference speed. Conventional sliding windows lose long-range history, while directly discarding past frames causes scene inconsistency.

### (2) Temporal and spatial compression

For historical frames $z_c$, the model first randomly samples one frame out of every 32 along the time dimension, then applies high-compression patchification. Frames farther from the current prediction are downsampled more:

- Frames from $t-1$ to $t-2$: downsampling ratio $(1,2,2)$.
- Frames from $t-3$ to $t-6$: downsampling ratio $(1,4,4)$.
- Frames from $t-7$ to $t-23$: downsampling ratio $(1,8,8)$.

Because the world is approximately continuous in space and time, this downsampling has relatively little effect on world-model performance while greatly relieving memory pressure.

### (3) Channel compression and linear attention

Historical frames $z_c$ are further patchified at compression ratio $(8,4,4)$ and compressed to 96 channels, yielding $z_{linear}$.

Inside a DiT block, after the current video tokens $z^l$ pass through cross-attention, the predicted frames $z_p^l$ are extracted and concatenated with $z_{linear}$ to obtain fused features $z_{fus}$.

To avoid standard attention's quadratic computation with sequence length, the model uses **linear attention** here.

## II. Anchor-Heavy KV Caching

Autoregressive models share a common problem: error accumulation. At block 100, the model refers to blocks 99 and 98. If block 99 contains a small error, block 100 continues from the distorted scene, eventually causing severe drift.

1. **Ordinary KV-cache logic**: when generating a new frame, the model “looks back” at preceding frames through the KV cache. It usually focuses on the most recent frames to maintain motion continuity.

2. **AHIS's core idea: favor older history over recent frames.**

   - The paper sets an attention window of **5 blocks**.
   - **Identity-anchor sink tokens**: the first 3 blocks, with the best quality and most faithful identity, are permanently locked in the cache instead of being evicted as time advances. They act like “ID photographs.”
   - **Rolling tokens**: the remaining 2 blocks store the most recently generated scenes and slide forward over time, maintaining current lip and action continuity.

3. **Why “anchor-heavy”?**

   - Conventional attention sinks, often used in LLMs, retain only a few tokens (such as the first word of the first sentence) to stabilize attention computation.
   - This paper deliberately gives identity anchors disproportionately large weight: 3 blocks versus 2.
   - **Effect**: whenever a frame is generated, this forces most attention onto the original, undistorted “perfect face” (high-fidelity representation), fundamentally suppressing imitation of subtle distortions in recent frames. A digital human's appearance therefore does not drift even after several minutes of continuous speech.

## III. Simulated Error Injection into Training Context

At inference, context consists of the model's own generated frames and inevitably includes errors. A model trained only to generate from real frames can easily enter OOD regions when errors occur. To address this, Kunlun Tech's Matrix-Game 3.0 injects simulated errors into training context, improving robustness to context errors in real situations.

1. **Grouped input**: divide video latents into the first $k$ frames (historical conditions) and the remaining $N-k$ frames (current noisy prediction targets).

2. **Error collection**: calculate the residual between the model's clean prediction $\hat{x}^t$ and real latent $x^t$:

$$
\delta = \hat{x}^t - x^t
$$

3. **Error injection**: store residuals in buffer $\mathcal{E}$, uniformly sample scalar $\gamma$ during training, and perturb historical latents:

$$
\tilde{x}^t = x^t + \gamma\delta
$$

4. **Condition injection and optimization**: inject discrete keyboard actions precisely through cross-attention and continuous mouse controls through self-attention. The final flow-matching objective is:

$$
\mathcal{L} = \mathbb{E}_{x,t,\epsilon,\delta}\left[\left\|\left(\epsilon - x^{k+1:N}\right) - v_{\theta}\left(x_t^{k+1:N}, t\mid \tilde{x}^{1:k}, c\right)\right\|_2^2\right]
$$

## IV. Camera-Aware Long-Term Memory

Matrix-Game 3.0 improves spatial consistency and stability in long autoregressive generation through a long-term-memory module.

Explicit long-term memory ensures that scene layout and details remain unchanged when a player “looks back” after roaming the virtual world for a long time.

- **Unified self-attention**: retrieved memory latents, recent past frames and current noisy frames occupy the same attention space and undergo joint spatiotemporal modeling in the same DiT. This avoids feature misalignment and slow convergence from external memory branches.
- **Camera-aware retrieval and encoding**: not all history is useful. The model retrieves relevant memory frames based on camera pose and field-of-view overlap, using relative Plücker coordinates for explicit geometric encoding to align the same scene across viewpoints. The memory path also uses the error-injection workflow above to reduce train–test mismatch.
- **Improved rotary positional embeddings (RoPE)**: to avoid periodic positional aliasing between distant memory frames and current frames, the model introduces independent perturbation coefficients $\epsilon_h$ for different attention heads:

$$
\hat{\theta}_h = \theta_{base}(1 + \sigma_{\theta}\epsilon_h)
$$

## References

- Wang, Z., Liu, Z., Li, J., Huang, K., Xu, B., Kang, F., An, M., Wang, P., Jiang, B., Wei, Y., Xietian, Y., Pei, J., Hu, L., Jiang, B., Xue, H., Wang, Z., Sun, H., Li, W., Ouyang, W., He, X., Liu, Y., Li, Y., & Zhou, Y. (2026). [Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory](https://arxiv.org/abs/2604.08995). arXiv:2604.08995.
- Chen, B., Monso, D. M., Du, Y., Simchowitz, M., Tedrake, R., & Sitzmann, V. (2024). [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://arxiv.org/abs/2407.01392). NeurIPS.
- Mao, X., Li, Z., Li, C., et al. (2025). [Yume-1.5: A Text-Controlled Interactive World Generation Model](https://arxiv.org/abs/2512.22096). arXiv:2512.22096.
- Li, W., Pan, W., Luan, P.-C., Gao, Y., & Alahi, A. (2025). [Stable Video Infinity: Infinite-Length Video Generation with Error Recycling](https://arxiv.org/abs/2510.09212). ICLR.
