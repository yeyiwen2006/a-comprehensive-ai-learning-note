---
title: "28.4 Distillation and Self Forcing for Video Diffusion Models"
chapter_title: "Multimodal Generation"
section_id: "28-04"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/28.多模态生成/28.4 视频生成扩散模型的蒸馏与Self Forcing.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 28.4 Distillation and Self Forcing for Video Diffusion Models

Although video diffusion models have very high visual fidelity, generating a short video usually takes 1 to 2 minutes because all frames require iterative denoising. This is impractical for real-time human–computer interaction and robotics. Improved distillation can turn large, slow models into very fast real-time models. Meanwhile, training video generation models for next-state prediction using only ground truth as context is highly prone to collapse. They must adapt autonomously to long rollouts and learn to correct errors along the way.

## I. Consistency Distillation

The original idea of consistency models (CM) is to train a student network $f_\theta(x_t,t)$ to map directly, in one step, from any time $t$ along a probability-flow ODE trajectory to its endpoint (clean data $x_0$). The mathematical constraint is:

$$
f_\theta(x_t,t)=f_\theta(x_{t'},t')
$$

This method is generally used for trajectory initialization during training: the student trains directly on teacher-generated trajectories and quickly learns denoising. The teacher often takes many steps to generate each frame (such as $N=48$), but the student cannot take so many steps if latency is to be reduced, so only key time steps are sampled. Because teacher and student take different numbers of steps, the target is changed here to real data rather than teacher-predicted noise.

To give the model fast generation and autoregressive (block-by-block) capabilities, the first step is to establish a strong foundation, teaching it to complete in very few steps the denoising that originally required many.

- Workflow:
  1. First run the large teacher model to generate complete $N=48$-step denoising trajectories $\{x_{t_j}\}_{j=0}^{N}$.
  2. Subsample only $k=4$ key time steps from this long trajectory.
  3. The causal student generator $g_\phi$ receives noisy latents and multimodal conditions $c$ (text, audio and reference images), and attempts to predict a completely clean video block $x_0$ directly within these 4 steps.
- Mathematical formula:

At this stage, the student learns by minimizing mean squared error between its prediction and the real noise-free data.

$$
\mathcal{L}_{ODE}=\mathbb{E}\left[\left\|g_\phi(x_{t_j},c)-x_0\right\|_2^2\right]
$$

## II. Trajectory Segmented Consistency Distillation (TSCD)

Making a network predict $x_0$ across a very long time interval is extremely difficult for high-dimensional, highly nonlinear data such as video, and can lead to local optima and blurry frames. TSCD, introduced from Hyper-SD, uses “divide and conquer.”

1. Segment the time domain: divide the complete trajectory $[0,T]$ into $k$ contiguous subintervals:

$$
[t_0,t_1],[t_1,t_2],\ldots,[t_{k-1},t_k],\quad t_0=0,\ t_k=T
$$

2. Segmentwise consistency: within each subinterval $[t_{i-1},t_i]$, the student is no longer forced to predict distant $x_0$, but only to maintain consistency within that interval. During generation (denoising), if the model is at $t_i$, it predicts only the lower endpoint of the current segment, the state $x_{t_{i-1}}$ closer to real data, rather than reaching for distant $x_0$.

3. Advantage: dividing a long, winding integration curve into shorter, relatively smooth curves greatly reduces the difficulty of fitting the diffusion process. At inference, the model need only jump between preset macro-level nodes to approximate the full trajectory accurately, achieving a strong speed–fidelity balance with fewer steps and a 4-fold speedup.

## III. Self Forcing and Distribution Matching Distillation (DMD)

### (1) Core idea

Consistency distillation based on teacher forcing is off-policy. Because the student's generation is autoregressive between blocks, any deviation in its trajectory takes it into an OOD region, making it highly unstable. We want on-policy distillation, or Self Forcing: building on the student's own generation, make its final generated-data distribution $p_\Phi(x)$ approach the real physical-world data distribution $p_{\mathrm{data}}(x)$ (which the teacher can approximate).

Research shows that directly applying MSE between student-generated and real data during Self Forcing readily causes training collapse, whereas distribution matching distillation solves this problem. The reason is explained later.

In probability and statistics, Kullback–Leibler (KL) divergence is a classic measure of distributional differences. We want to minimize $\mathcal{D}_{\mathrm{KL}}(p_\phi\|p_{\mathrm{data}})$.

The key point is that, within the mathematical framework of score-based diffusion models, the gradient of this loss with respect to generated data, used to narrow the distributional gap through gradient descent, equals the difference between the two distributions' scores:

$$
\begin{aligned}
\nabla_x\mathcal{D}_{\mathrm{KL}}(p_\phi\|p_{\mathrm{data}})
&=\nabla_x\log p_\phi(x)-\nabla_x\log p_{\mathrm{data}}(x)
\end{aligned}
$$

The second term is the gradient of the log distribution of real denoised data. By diffusion-model principles, the teacher learns the real-data distribution, so this term approximately equals the teacher's noise prediction. The first term is the gradient of the log distribution of student-predicted denoised data. A “critic network” learns the student's output distribution; noise is added to samples from it, and its noise prediction provides this term. The gradient is then computed to update student parameters.

### (2) Workflow

1. The critic learns the student's denoising behavior

The critic workflow is:

1. Student generator $g_\phi$ produces predicted video frames $\hat{x}_0$ from Gaussian noise $z$ and multimodal conditions $c$: $\hat{x}_0=g_\phi(z,0,c)$.
2. The system artificially re-noises the generated $\hat{x}_0$ to time $\tau$, obtaining $x_\tau$.
3. Critic $s_\psi$ acts as a “judge,” closely tracking the student's changing distribution and attempting to denoise $x_\tau$ back to $\hat{x}_0$.

Mathematically:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{critic}}
&=\mathbb{E}_{\tau}\left[\left\|s_\psi(x_\tau,\tau,c)-\hat{x}_0\right\|_2^2\right]
\end{aligned}
$$

2. Update the student using gradients from the teacher–critic difference

We now have a continuously generating “student” and a “judge” that tracks its level. The final step is to introduce a frozen “expert teacher score network” $s_\theta$, possessing fully correct knowledge, to correct the student's errors.

Note: introducing multimodal conditions $c$ makes this computation very prone to collapse, which is why the paper stresses high-quality training data and aggressive learning-rate schedules.

The alternating gradient-update workflow is:

1. One-step generation: student generator $g_\phi$ directly generates predicted data $\hat{x}_0$ from noise and conditions.
2. Random noise addition: artificially re-noise $\hat{x}_0$ to time $\tau$, obtaining noisy data $\hat{x}_\tau$ (also written as $x_\tau$ in the document's prose).
3. Compute the score difference: feed the same noisy data $\hat{x}_\tau$ and condition $c$ into both the “expert teacher” $s_\theta$ and the “judge” critic $s_\psi$.
4. Immediate backpropagation: compute the difference between their scores (gradients). This provides a precise penalty signal, immediately backpropagated through the chain rule to update student generator $g_\phi$. Mathematically:

$$
\begin{aligned}
\nabla_\phi\mathcal{L}_{\mathrm{DMD}}
&=\mathbb{E}_{\tau}\left[
\frac{\partial\hat{x}_0}{\partial\phi}
\frac{s_\psi(\hat{x}_\tau,\tau,c)-s_\theta(\hat{x}_\tau,\tau,c)}{\tau}
\right]
\end{aligned}
$$

5. Alternating updates: DMD training continually alternates between updating the critic and updating the student generator.

This is on-policy because noise is added to the student's own generation, which then serves as data for both teacher and critic.

### (3) Why does DMD work well while MSE causes collapse?

MSE's critical flaw (regression to the mean): if the student must generate an image in only 1 step, its trajectory necessarily differs from that of a multistep teacher. Given the same text, for example, the teacher draws a yellow cat and the student a black cat. Both are reasonable, but pixel-level MSE regards the student as terribly wrong and forces it toward the average of yellow and black, eventually producing a blurry gray mass.

DMD's solution (distribution-level alignment): it abandons strict pixel-level alignment and instead uses KL divergence to measure the overall difference between generated and real distributions. Its core gradient is approximately:

$$
\begin{aligned}
\nabla_\theta\mathcal{L}_{\mathrm{DMD}}
&\approx
\mathbb{E}\left[
w(t)
\left(\hat{\epsilon}_{\mathrm{fake}}(\hat{x}_t,t)-\hat{\epsilon}_{\mathrm{real}}(\hat{x}_t,t)\right)
\frac{\partial\hat{x}_t}{\partial\theta}
\right]
\end{aligned}
$$

- $\hat{\epsilon}_{\mathrm{real}}$: the score computed by the frozen, powerful teacher representing the real-data distribution.
- $\hat{\epsilon}_{\mathrm{fake}}$: the score computed by a model specifically fitted to the student's fake data.

DMD's logic is that, as long as the student's output (such as a black cat) follows the real-world probability distribution, the teacher also considers it reasonable, the difference between $\hat{\epsilon}_{\mathrm{real}}$ and $\hat{\epsilon}_{\mathrm{fake}}$ is small, and no penalty gradient is produced.

DMD also has drawbacks. Beyond high compute costs and the need for a powerful teacher, instability must be considered:

MSE's greatest advantage is that it is “straightforward and stable.” For any pixel, the Euclidean distance between prediction and truth is a very smooth convex approximation (locally). This lets the loss decrease stably during distributed training on huge clusters and is a foundation of scaling laws.

Distribution matching, whether GAN loss based on adversarial networks or DMD based on score matching, is a dynamic game.

- Instability: the student and the fake model (or discriminator) evaluating distributional differences pull against each other during training. If one evolves too quickly, gradients can instantly explode or vanish, bringing the expensive training cluster to a halt.
- Mode collapse: to minimize global distributional differences, the model often takes shortcuts. It finds that repeatedly producing a few high-quality scenes it handles particularly well (such as always generating slow-motion static landscapes) yields high distributional-evaluation scores. Generation diversity is lost.

### (4) Gradient truncation

In standard autoregressive generation, a video sequence of length $N$ with $T$ diffusion-denoising steps per frame has a full computation-graph depth of $N\times T$.

Standard backpropagation through time (BPTT) requires all forward activations from these $N\times T$ steps to be retained in GPU memory for gradient computation. Memory usage then explodes exponentially with sequence length and diffusion steps. Even top-end cards with 80GB of memory (such as H100) cannot support complete BPTT for a few seconds of video.

Stochastic gradient truncation therefore “prunes” the graph, keeping memory usage within a constant, acceptable range while allowing the model to learn error tolerance.

There are two specific truncation methods:

1. Truncation across time steps

Gradients from generation step i propagate back at most to step i-m, preventing excessive graph depth.

2. Truncation across denoising steps

Even after gradients between frames are cut, the $T$ denoising steps within a frame (even $T=4$ in a few-step model) still consume substantial memory.

At every training iteration, stochastic truncation randomly samples a denoising step $s_i\in\{1,2,\ldots,T\}$ for each frame $i$. Backpropagation computes gradients only from step $T$ to step $s_i$, discarding the graph of earlier denoising steps. This saves memory and introduces randomness with a regularization effect similar to dropout.

### (5) Engineering strategies

Multimodality makes DMD training extremely fragile. Possible improvements are:

- Improve multimodal conditioning data: use a large visual model (such as Qwen-Image) to improve reference-image quality, and Qwen2.5-VL to strengthen descriptions of dynamic features in text prompts, preventing collapse caused by low-quality data.
- Fully converged ODE initialization: unlike earlier text-to-video distillation, this model must train to full convergence in the ODE stage (20k steps), providing a strong starting point for the sensitive generator–critic game.
- Aggressive optimization schedules: because the effective learning window for multimodal distillation is short (degradation usually starts after a few hundred steps), the team doubles the learning rate and greatly increases the teacher's classifier-guidance (CFG) ratio, forcing the model to learn difficult alignment tasks such as lip synchronization within the window.

## IV. DMD in Flow-Matching Models

The objective is to minimize KL divergence between the student's generated distribution $p_{\mathrm{fake}}$ and the real-data distribution $p_{\mathrm{real}}$. The core gradient with respect to generator parameters $\theta$ is:

$$
\begin{aligned}
\nabla_\theta\mathcal{L}_{\mathrm{DMD}}
&=
\mathbb{E}_{z,\epsilon,t}\left[
\omega(t)
\left(v_{\mathrm{teacher}}(x_t,t)-v_{\mathrm{fake}}(x_t,t)\right)
\nabla_\theta G_\theta(z)
\right]
\end{aligned}
$$

The variables mean:

- $z\sim\mathcal{N}(0,I)$ is the initial noise fed into the student generator.
- $G_\theta(z)$ is the student-generated image (a fake sample).
- $\epsilon\sim\mathcal{N}(0,I)$ is reference noise used to construct intermediate states.
- $t\in[0,1]$ is the time step, usually sampled uniformly.
- $x_t=tG_\theta(z)+(1-t)\epsilon$ is the intermediate state interpolated at time $t$ along the optimal-transport path.
- $v_{\mathrm{teacher}}(x_t,t)$ is the true vector field predicted at this state by a pretrained teacher flow-matching model (such as SD3 or Flux), representing the direction toward the real-data manifold.
- $v_{\mathrm{fake}}(x_t,t)$ is a jointly trained fake vector-field estimator fitted to the vector field of current student-generated trajectories.
- $\omega(t)$ is a time-dependent weighting function.

To understand this expression, we need to derive the mathematical relationship between the flow-matching vector field $v_t(x_t)$ and marginal score $\nabla_{x_t}\log p_t(x_t)$.

In optimal-transport conditional flow matching (OT-CFM), the probability path is a straight interpolation from noise to data:

$$
x_t=t x_1+(1-t)x_0
$$

Here, $x_0\sim\mathcal{N}(0,I)$ is the Gaussian prior and $x_1\sim p_{data}$ the target data. Given endpoint $x_1$, the conditional distribution of $x_t$ is Gaussian:

$$
p_t(x_t|x_1)=\mathcal{N}\left(x_t;t x_1,(1-t)^2I\right)
$$

By conditional Gaussian properties, the marginal score can be expressed as the expectation of conditional scores:

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\mathbb{E}_{x_1\sim p(x_1|x_t)}
\left[
\nabla_{x_t}\log p_t(x_t|x_1)
\right]
\end{aligned}
$$

Differentiating the conditional distribution gives:

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t|x_1)
&= -\frac{x_t-tx_1}{(1-t)^2}
\end{aligned}
$$

Substitution into the expectation gives the marginal score:

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\frac{t\mathbb{E}[x_1|x_t]-x_t}{(1-t)^2}
\end{aligned}
$$

In flow matching, the target vector field $v_t(x_t)$ is the conditional expectation of the path derivative $\dot{x}_t=x_1-x_0$. Since $x_0=\frac{x_t-tx_1}{1-t}$:

$$
\begin{aligned}
\dot{x}_t=x_1-\frac{x_t-tx_1}{1-t}
&=
\frac{x_1-x_t}{1-t}
\end{aligned}
$$

Taking the conditional expectation given $x_t$:

$$
\begin{aligned}
v_t(x_t)=\mathbb{E}[\dot{x}_t|x_t]
&=
\frac{\mathbb{E}[x_1|x_t]-x_t}{1-t}
\end{aligned}
$$

We can solve this for the posterior expectation of data $x_1$:

$$
\mathbb{E}[x_1|x_t]=x_t+(1-t)v_t(x_t)
$$

Substitute $\mathbb{E}[x_1|x_t]$ back into the score formula:

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\frac{t(x_t+(1-t)v_t(x_t))-x_t}{(1-t)^2}
\\
&=
\frac{(t-1)x_t+t(1-t)v_t(x_t)}{(1-t)^2}
\\
&=
\frac{t v_t(x_t)-x_t}{1-t}
\end{aligned}
$$

We have thus obtained the relationship between the score and the flow-matching velocity field.

Let the real-data score be $s_{real}$ and the student-data score $s_{fake}$. We know:

$$
\begin{aligned}
\nabla_\theta\mathcal{L}_{DMD}
&\propto
\mathbb{E}\left[
w(t)
\left(s_{\mathrm{real}}(x_t,t)-s_{\mathrm{fake}}(x_t,t)\right)
\frac{\partial x_t}{\partial\theta}
\right]
\end{aligned}
$$

From the derivation above:

$$
\begin{aligned}
s_{\mathrm{real}}(x_t,t)-s_{\mathrm{fake}}(x_t,t)
&=
\frac{t v_{\mathrm{real}}(x_t,t)-x_t}{1-t}
\\
&\quad-
\frac{t v_{\mathrm{fake}}(x_t,t)-x_t}{1-t}
\\
&=
\frac{t}{1-t}\left(v_{\mathrm{real}}(x_t,t)-v_{\mathrm{fake}}(x_t,t)\right)
\end{aligned}
$$

Substituting this result into the gradient formula and expanding the chain rule $\nabla_\theta x_t=t\nabla_\theta G_\theta(z)$ yields the core DMD gradient for flow matching. The conversion coefficient $\frac{t}{1-t}$ and chain-rule factor $t$ are ultimately absorbed into $\omega(t)$.

## V. Checkpointed Self Forcing

### (1) The problem with Self Forcing

Conventional Self Forcing requires student and teacher to have the same context length. Solaris, however, introduces a sliding window during student-video generation so the student can benefit from a long-context teacher.

Direct backpropagation with a sliding window causes severe memory problems:

- Graph redundancy: each generated frame advances the window by one step, producing a new context window. Step 1, for example, uses frames $1:L_s$, and step 2 uses $2:L_s+1$.
- Memory explosion: backpropagation in frameworks such as Jax retains all overlapping windows simultaneously. With student context length $L_s$ and total generation steps $L_t$, this redundancy costs up to $O(L_t\cdot L_s)$ memory.

### (2) The core idea of Checkpointed Self Forcing

During training, replace sliding-window autoregressive inference at every step with parallelization and an attention mask that simulates the sliding window. This alone cannot guarantee autoregressivity, so Checkpointed Self Forcing uses “autoregressive generation without gradients first, then parallel training with gradients.” The workflow is:

1. Gradient-free autoregressive rollout

This stage simulates autoregressive inference, collecting “clean historical context” and “current noisy targets,” while completely disabling graph construction in code to save substantial GPU memory.

- Global initialization: set teacher context length $L_t$ (the total frames to generate) and student context length $L_s$ (the sliding-window size). Initialize empty lists $X_0$ (clean estimated frames) and $X_s$ (noisy transitional frames), and an empty KV cache.
- Randomly sample truncation step $s$: uniformly select $s$ from all denoising time steps $\{t_1,\ldots,t_T\}$. This $s$ is the “stop denoising” target for this training round, corresponding to the previously mentioned $\sigma_{stop}$.
- Frame-by-frame outer loop: for each frame $i$ from 1 to $L_t$:
  - Inject pure noise: initialize pure random noise $x_{t_T}^{i}\sim\mathcal{N}(0,I)$ for frame $i$.
  - Inner denoising loop: denoise backward step by step from $T$ to $s$.
  - If the current step equals truncation step $s$, the desired intermediate noise level has been reached. Store the current noisy state $x_{t_s}^{i}$ in $X_s$. Then model $G_\theta$ predicts the corresponding clean frame $\hat{x}_0^i$ from this noisy state and the current KV cache, and stores it in $X_0$.

Result: two sequences of length $L_t$, a clean historical-frame sequence $X_0=[\hat{x}_0^1,\ldots,\hat{x}_0^{L_t}]$ and a noisy transitional-frame sequence $X_s=[x_{t_s}^1,\ldots,x_{t_s}^{L_t}]$. No backpropagation computation graph is held in GPU memory at this point.

2. Data processing

Completely detach gradients: explicitly call `stop_grad()` again on $X_s$ and $X_0$, ensuring that they enter the next step as constant inputs.

Concatenate sequences: concatenate the length-$L_t$ clean sequence $X_0$ and length-$L_t$ noisy sequence $X_s$ along the sequence dimension, forming $X_{in}=[X_0,X_s]$. The total input length doubles to $2L_t$.

3. Masked parallel recomputation and backpropagation

Parallel recomputation now begins, with each token denoising one step from the previously sampled noise level. A special mask simulates sliding-window attention and can be described by four quadrants:

- Bottom right (noisy queries attending to noisy keys): noisy frame $x_{t_s}^i$ sees only itself, not other noisy frames or past noisy frames, producing a diagonal.
- Bottom left (noisy queries attending to clean keys): noisy frame $x_{t_s}^i$ can see past clean frames $\hat{x}_0$, but must strictly obey causality and the window limit, seeing only $\hat{x}_0^{i-L_s:i-1}$ to prevent information leakage across time.
- Top left (clean queries attending to clean keys): clean frames $\hat{x}_0$ see past clean frames under ordinary causal sliding-window rules.
- Top right (clean queries attending to noisy keys): entirely black, or completely forbidden. Clean frames must never see information from noisy frames, preventing information leakage across time.

Backpropagation then needs only one $O(L_t)$ computation graph in GPU memory.

## VI. In-Loop Self-Distillation (ILSD)

In “ELT: Elastic Looped Transformers for Visual Generation,” Google DeepMind proposes in-loop self-distillation (ILSD) to freely adjust the number of diffusion-model denoising loops. Every intermediate output becomes meaningful instead of remaining meaningless noise before the preset maximum number of steps, accommodating different hardware capabilities (for example, more steps for high-resolution images in the cloud and fewer for ordinary images on-device):

- Teacher path: run the full maximum number of loops $L_{\max}$, producing the most mature, highest-fidelity internal representation.
- Student path: at every training iteration, uniformly sample an intermediate loop count $L_{\mathrm{int}}$, where $L_{\min}\le L_{\mathrm{int}}\lt L_{\max}$, and extract the output there.

Training aims not only to fit the final output to real labels but also to make the “intermediate-step student” imitate the “fully completed teacher.” The joint ILSD loss $\mathcal{L}^{ILSD}_{\Theta}$ combines three terms:

- The first is the teacher's ($L_{\max}$) ground-truth loss for real data $y$.
- The second is the student's ($L_{\mathrm{int}}$) ground-truth loss.
- The third is distillation loss, pulling student output closer to teacher output. $\mathrm{sg}$ means stop-gradient is applied to the teacher prediction in this term.
- $\lambda$ is a curriculum weight that decreases linearly from 1 to 0 during training. This forces the shared-parameter block to compress complex transformations into earlier loops.

## References

- Salimans, T., & Ho, J. (2022). [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512). ICLR.
- Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). [Consistency Models](https://arxiv.org/abs/2303.01469). ICML.
- Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., & Park, T. (2024). [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828). CVPR.
- Ren, Y., Xia, X., Lu, Y., et al. (2024). [Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis](https://arxiv.org/abs/2404.13686). NeurIPS.
- Huang, X., Li, Z., He, G., Zhou, M., & Shechtman, E. (2025). [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009). arXiv:2506.08009.
- Savva, G., Michel, O., Lu, D., Waiwitlikhit, S., Meehan, T., Mishra, D., Poddar, S., Lu, J., & Xie, S. (2026). [Solaris: Building a Multiplayer Video World Model in Minecraft](https://arxiv.org/abs/2602.22208). arXiv:2602.22208.
- Goyal, S., Agrawal, S., Anil, G. G., Jain, P., Paul, S., & Kusupati, A. (2026). [ELT: Elastic Looped Transformers for Visual Generation](https://arxiv.org/abs/2604.09168). arXiv:2604.09168.
