---
title: "28.2 Main Training Stages of Video Generation Models"
chapter_title: "Multimodal Generation"
section_id: "28-02"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/28.多模态生成/28.2 视频生成模型的主要训练阶段.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 28.2 Main Training Stages of Video Generation Models

## I. Pretraining

Pretraining video generation models or generative world models generally uses vast amounts of unsupervised video data (which may now also include audio in industry), with or without text or image labels. The model predicts the next state from preceding video states, text or image prompts, learning world representations and physical laws in the process. Specific tasks include Text-to-Video-Audio, Image-to-Video-Audio, Video-Audio-to-Video-Audio (continuing a video sequence), Text-to-Video, Image-to-Video, and Video-to-Video.

When training joint video and audio generation, the model architecture changes as follows:

When solving the vector field described above, the network uses a dual-branch structure. At every Transformer layer, video and audio tokens not only compute self-attention but also exchange information through cross-attention.

During denoising at any time step $t$:

1. The video branch “listens” to the current audio's rhythm and semantics, adjusting a character's mouth shape or movement amplitude in the next frame.

2. The audio branch “observes” physical collisions or environmental changes in the scene, synthesizing precisely synchronized sound effects or echoes.
During training, the proportion of noise perturbation can be increased for high-resolution and long videos to improve convergence stability.

Later in training, curriculum learning can gradually extend video length. The model gradually increases the flow shift to give more weight to high-noise time steps, helping stabilize global scene structure over very long durations and reduce drift. An optical-flow evaluator selects highly dynamic data to specifically improve motion amplitude and smoothness.

## II. Supervised Fine-Tuning / Mid-Training

### (1) Fine-tuning for action dynamics

1. Dataset preparation

Narrative caption: a global semantic prompt combining environmental information, camera motion and temporal evolution. During the forward pass, these text features are injected as conditions through the DiT architecture's cross-attention layers.

Scene-static caption: deliberately ignores all camera motion and action information. This annotation forces the model to separate “what the scene looks like” from “how the scene moves” during learning, allowing it to learn physical dynamics driven by control signals rather than text alone.

Dense temporal caption: divides a long video into time intervals and annotates specific events. It supports temporal alignment training, helping the model understand and predict very fine-grained temporal actions.
Dense temporal captions are equivalent to adding a description of action $a_t$ to the context after the video for each time interval $s_t$, helping the model learn $p(s_{t+1}\mid s_t,a_t)$.

2. Fine-tuning method

At this stage, user action commands are injected into the model to support interaction.

Action representation and injection: continuous camera rotation (Plücker embeddings) and discrete keyboard input (multi-hot W, A, S, D encodings) are concatenated along the channel dimension. The combined action signal is injected into DiT blocks through AdaLN, dynamically modulating feature generation.

Parameter-efficient fine-tuning: to prevent catastrophic forgetting during the learning of interaction, namely the loss of the original high-fidelity image quality, the main DiT-block parameters are frozen. Only the newly added action-adapter layers are fine-tuned, including action-embedding projections and AdaLN parameters.

### (2) Self Forcing (self-rollout training)

Pretraining is inherently off-policy, so long autoregressive generation still accumulates errors. The model can be conditionally trained to read its own generated historical frames through the KV cache (in contrast to the “ground-truth” history read during pretraining) and predict the next frame, forcing it to recover from its own errors and artifacts.

### (3) Fine-tuning for different video styles

Fine-tune on extremely high-quality, human-verified data across hundreds of carefully defined categories based on motion type and visual style. Multiple specialized models can first be fine-tuned separately and then merged to balance visual fidelity with prompt controllability.

In multi-style video generation, mixing photorealistic, anime, 3D-rendered and other data in one dataset for supervised fine-tuning (SFT) of a single model can readily create conflicting parameter-optimization directions, causing catastrophic forgetting or style contamination.

Specialized training: the team first fine-tunes multiple specialized models separately on carefully constructed subsets for different visual styles and motion types.

Weight merging: because all models start from the same pretrained base, their parameter distributions lie in the same loss basin. Their weight matrices can therefore be merged algebraically without additional training compute. Common operations include:

1. Weight averaging: take the arithmetic mean of the parameter matrices of multiple expert models.

2. Spherical linear interpolation (SLERP): interpolate on a high-dimensional hypersphere, preserving the nonlinear geometric characteristics of large matrices better than direct averaging.

3. Task arithmetic: extract each expert's task vector (expert weights minus base weights) and add scaled versions back to the base model.

Through merging, the final single model inherits the visual fidelity and prompt controllability of different experts while reducing overfitting.
On spherical linear interpolation (SLERP): (applicable only to fully fine-tuned large models, not to LoRA-fine-tuned large models)

To understand SLERP, first compare it with the simplest linear interpolation (LERP, or weight averaging/model soup). If model-weight tensors are treated as two points in high-dimensional space, LERP connects them with a straight line. Its critical drawback is that the middle of the line “passes through” the sphere's interior, severely reducing the interpolated vector's magnitude (interpretable as weight energy or variance) and thereby losing the model's original representational ability.

The mathematical foundation of diffusion models is the isotropic standard normal distribution $\mathcal{N}(0,I)$. Geometrically and topologically, this extremely high-dimensional Gaussian distribution can be viewed as a high-dimensional hypersphere.

With linear interpolation, noise variance in intermediate frames decreases. The model regards this as “invalid, contaminated” noise and decodes blurry or structurally broken intermediate frames. Only SLERP, moving along the Gaussian hypersphere, ensures that the transitional noise of every frame satisfies $\mathcal{N}(0,I)$, allowing the diffusion model to denoise frame by frame into smooth transition animations.

In diffusion models, reduced weight variance directly causes a sharp decrease in network activation magnitudes. In generated videos or images, this appears as gray-looking frames, lower contrast and lost high-frequency detail, which can be called “model collapse” in computer vision.

Suppose two unit vectors $v_1$ and $v_2$ have angle $\Omega$ between them. Introduce an interpolation coefficient $t\in[0,1]$ and seek an interpolated vector $v(t)$ in their two-dimensional plane such that $v(t)$ is also a unit vector, the angle from $v_1$ to $v(t)$ is $t\Omega$, and the angle from $v(t)$ to $v_2$ is $(1-t)\Omega$. The SLERP formula is:

$$
v(t)=\frac{\sin((1-t)\Omega)}{\sin\Omega}v_1+\frac{\sin(t\Omega)}{\sin\Omega}v_2
$$

In actual AI engineering code (such as the widely used `mergekit` tool), model matrices may contain billions of parameters. The formula cannot simply be applied to arbitrary matrices; a rigorous engineering procedure is required. Consider merging model A (weights $\theta_A$) and model B (weights $\theta_B$):

1. Tensor matching for shape and vocabulary alignment: scan both network architectures before merging. If shapes or vocabularies do not match, pad with empty tensors to force alignment to the same dimensions.

2. Flattening: extract corresponding tensor layers and flatten them into one-dimensional vectors $\mathbf{v}_A$ and $\mathbf{v}_B$.

3. Magnitude extraction and normalization: calculate their original Euclidean norms $\|\mathbf{v}_A\|$ and $\|\mathbf{v}_B\|$, then divide each vector by its norm to obtain direction vectors $\hat{\mathbf{v}}_A$ and $\hat{\mathbf{v}}_B$ on the unit hypersphere.

4. Angle cosine calculation using a dot product: compute $d=\hat{\mathbf{v}}_A\cdot\hat{\mathbf{v}}_B$, namely $\cos\Omega$.

5. Fallback check for protection against extreme values and collinearity: implementations usually set a threshold, such as `DOT_THRESHOLD = 0.9995`. If the dot product is extremely close to 1, the two models are almost parallel in that parameter layer, $\Omega\approx 0$, and $\sin\Omega$ approaches 0, causing division by zero in the formula. The algorithm then falls back to ordinary LERP.

6. Apply SLERP: calculate $\Omega=\arccos(d)$ and substitute the specified merging ratio $t$ into the SLERP formula to obtain the new direction-interpolated unit vector $\hat{\mathbf{v}}_{\mathrm{new}}$.

7. Magnitude scaling: also linearly interpolate the original magnitudes:

$$
M=(1-t)\|\mathbf{v}_A\|+t\|\mathbf{v}_B\|
$$

Then multiply the new magnitude into the direction vector:

$$
\mathbf{v}_{\mathrm{new}}=\hat{\mathbf{v}}_{\mathrm{new}}\times M
$$

8. Reshape: fold the one-dimensional $\mathbf{v}_{\mathrm{new}}$ back into the original multidimensional matrix shape and replace the original model's weights.

## III. RL

### (1) RLHF in video generation

1. Clean-video prediction: unlike language models that output tokens, each diffusion-model sampling trajectory is a continuous evolution of noise. During training iterations, the model simulates the complete video-inference pipeline, using the current denoising policy to predict a noise-free generated video directly.

2. Multidimensional reward evaluation: send the predicted video to three separate reward models: a base model (evaluating image–text alignment and structure), a motion model (penalizing artifacts and rewarding smoothness), and an aesthetic model (extracting visual-quality scores from keyframes).

3. Composite-gradient backpropagation: linearly combine the reward-model scores. Through critic-free GRPO, use relative advantages within a group generated for the same prompt to maximize the composite reward directly and backpropagate gradients to the diffusion model.

4. Multiple rounds of dynamic interaction: to prevent reward hacking, such as making the scene completely static for a high aesthetic score, the generative and reward models undergo multiple rounds of iterative learning, steadily improving alignment with human preferences.

### (2) Learning physical laws through preference optimization in ABot-PhysWorld

Amap's ABot-PhysWorld shifts the optimization objective from pixel similarity to physical consistency through two core components:

Proposer module: lists the physical rules for the current task, specifying what is allowed and what is strictly forbidden.

Scorer module: scores multiple model-generated results frame by frame.

Diffusion-DPO then improves physical compliance using preference data: physically correct results are preferred samples, while physically incorrect ones are dispreferred. Through repeated optimization, the model gradually learns “which actions do not violate physics.” For example, given end-effector poses and gripper states, it can predict future spatiotemporal dynamics rather than merely producing pixel-level visual resemblance.

## IV. Model Distillation and Self Forcing

For models trained as described above, low latency is often essential when interacting with the physical world. Fewer diffusion steps are therefore needed, requiring model distillation. Sometimes a diffusion model with bidirectional attention serves as the teacher and an autoregressive model with causal attention serves as the student, giving the student a more global view of the future.

Moreover, video generation models or generative world models are pretrained using ground truth as context, an off-policy process. During long rollouts, a small error can easily push the model into an OOD region and cause collapse. During training, we therefore need the model to generate using its own trajectories as context and learn autonomous error correction, namely Self Forcing. Research shows that Self Forcing needs distribution matching distillation (DMD) for stable training.

For details, see the section on diffusion-model distillation and Self Forcing.

## V. Generative Adversarial Optimization

Video generation models or generative world models aim to “simulate the real world”; one approach is to make the simulation “more realistic.”

Adversarial optimization: because pure DMD distillation lacks direct supervision from real data, the student tends to inherit teacher biases. To improve generation quality, a GAN classification head $D(\cdot)$ is attached to the fake-score network for adversarial training. Its objectives are the generator loss $\mathcal{L}_G$ and discriminator loss $\mathcal{L}_D$:

$$
\begin{aligned}
\mathcal{L}_G
&=
\mathbb{E}_{p(\tilde{x})}
\left[
f\left(1-D(\mu_{\mathrm{fake}}(\tilde{x}_t,t,a))\right)
\right]
\end{aligned}
$$

$$
\begin{aligned}
\mathcal{L}_D
&=
\mathbb{E}_{p(x)}
\left[
f\left(D(\mu_{\mathrm{fake}}(x_t,t,a))\right)
\right]
\\
&\quad -
\mathbb{E}_{p(\tilde{x})}
\left[
f\left(1-D(\mu_{\mathrm{fake}}(\tilde{x}_t,t,a))\right)
\right]
\end{aligned}
$$

Here, $f$ is the softplus function, and $p(x)$ and $p(\tilde{x})$ are the real- and synthetic-video distributions, respectively.

## VI. SOAR: Self-Correction Training

HY-SOAR, proposed by Tencent's Hunyuan team, lets the model learn self-reflection and correction during denoising. SFT only teaches the model to handle “ideal trajectories,” whereas at inference time the model follows its own trajectory. If early denoising deviates, subsequent states enter regions never seen during training. RL, meanwhile, compresses rich trajectory-level information in the data into a scalar reward, losing many signals that could correct intermediate steps. When data quality is high enough, utilization becomes the bottleneck. RL discounts this utilization, and SOAR aims to recover what was lost.

SOAR workflow:

1. Perform one gradient-free forward inference step on a real sample to simulate a deviation the model itself might produce.

2. Re-noise the deviated state to construct an auxiliary training point.

3. Compute an analytical correction target anchored to the original sample.

SOAR does not aim to replace RL, but to provide a more stable starting point for it. A core challenge in current RL is that the base model's generation trajectories are not yet stable enough. Driving exploration directly with rewards can lead to excessive changes in unstable regions, improving one metric while other dimensions collapse. SOAR first raises the baseline of trajectory stability; RL can then explore preferences within a safer range for style adjustment and quality optimization.

## VII. Data Processing

High-quality training data underpins convergence of these models. Data cleaning is usually a highly automated filtering pipeline:

1. Shot segmentation: use inter-frame difference detection or pretrained detectors to split raw long videos into coherent clips of up to 12 seconds.

2. Visual-obstruction repair: combine heuristic rules and object detectors to identify and adaptively crop noisy regions such as watermarks and subtitles.

3. Semantic deduplication: use an internal video-representation model to extract feature vectors and cluster them. Within highly similar clusters, retain only the sample with the highest aesthetic score.

4. Long-tail distribution rebalancing: measure video probability distributions across actions, styles, resolutions and other dimensions. Downsample redundant head categories and apply probability weighting to tail categories, ensuring a balanced joint distribution.

## References

- Bruce, J., Dennis, M., Edwards, A., et al. (2024). [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). ICML.
- Hu, A., Russell, L., Yeo, H., et al. (2023). [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080). arXiv:2309.17080.
- Gao, Y., Guo, H., Hoang, T., et al. (2025). [Seedance 1.0: Exploring the Boundaries of Video Generation Models](https://arxiv.org/abs/2506.09113). arXiv:2506.09113.
- Chen, Y., Chen, R., Huo, D., et al. (2026). [ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment](https://arxiv.org/abs/2603.23376). arXiv:2603.23376.
- Lin, W., Chen, R., Liu, B., et al. (2025). [ContentV: Efficient Training of Video Generation Models with Limited Compute](https://arxiv.org/abs/2506.05343). arXiv:2506.05343.
- Zhang, G., Zhou, Z., Hu, T., et al. (2025). [UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](https://arxiv.org/abs/2511.03334). arXiv:2511.03334.
- Wortsman, M., Ilharco, G., Gadre, S. Y., et al. (2022). [Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time](https://arxiv.org/abs/2203.05482). ICML.
- Ilharco, G., Ribeiro, M. T., Wortsman, M., et al. (2023). [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089). ICLR.
- Huang, X., Li, Z., He, G., Zhou, M., & Shechtman, E. (2025). [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009). NeurIPS.
- Yin, T., Gharbi, M., Zhang, R., et al. (2024). [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828). CVPR.
- Yin, T., Gharbi, M., Park, T., et al. (2024). [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867). NeurIPS.
- Lin, S., Xia, X., Ren, Y., et al. (2025). [Diffusion Adversarial Post-Training for One-Step Video Generation](https://arxiv.org/abs/2501.08316). ICML.
- Qin, Y., Wang, L., Fei, H., et al. (2026). [SOAR: Self-Correction for Optimal Alignment and Refinement in Diffusion Models](https://arxiv.org/abs/2604.12617). arXiv:2604.12617.
- Shoemake, K. (1985). [Animating Rotation with Quaternion Curves](https://doi.org/10.1145/325334.325242). SIGGRAPH, 245–254.
