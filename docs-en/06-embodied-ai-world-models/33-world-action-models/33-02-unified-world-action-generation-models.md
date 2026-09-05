---
title: "33.2 Unified World–Action Generation Models"
chapter_title: "World–Action Models"
section_id: "33-02"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/33.世界-动作模型/33.2 世界-动作统一生成模型.docx"
status: "manually rebuilt and checked against Word"
ocr: "text/formula images manually transcribed; visual figures retained as cropped public assets"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 33.2 Unified World–Action Generation Models

## I. From LingBot-VA to LingBot-VA 2.0: Predict States First, Then Obtain Actions

### (1) Model Architecture

1. Sequential architecture and MoT

Visual observations are first compressed by a causal video VAE into compact latents $z_t\in\mathbb{R}^{N\times d}$, while action vectors are projected into tokens $a_t$ of the corresponding dimension.

At each autoregressive step, the model first predicts latent states for the next $K$ video frames from observation and action histories:

$$
z_{t+1:t+K}\sim p_\theta\!\left(\,\cdot\mid z_{\leq t},a_{\lt t},c_t\right).
$$

The motion-dynamics model then generates specific actions using predicted future visual states, historical observations and actions, and task conditions:

$$
a_{t:t+K-1}\sim q_\phi\!\left(\,\cdot\mid\hat{z}_{t+1:t+K},z_{\leq t},a_{\lt t},c_t\right).
$$

State and action prediction occur in the same model using an MoT architecture: they interact in shared attention layers but have separate parameters in FFN/MoE layers. At output, the model can alternate video and action tokens in the sequence.

Because vision contains more information than actions, their design is often asymmetric in two ways:

(1) Parameters: In LingBot-VA 2.0, for example, the action expert has fewer parameters, while the visual expert uses a relatively sparse MoE architecture, activating only the top eight of 128 experts during inference.

(2) Hidden dimensions: In FFN/MoE layers, video has a larger hidden dimension than actions because it contains more information. To align hidden dimensions in attention layers, downsampling or upsampling is generally applied during forward propagation between layers.

2. Backbone

LingBot-VA uses the video-generation model Wan2.2 as its pretrained backbone and post-trains it. LingBot-VA 2.0 pretrains a diffusion Transformer from scratch on embodied data, making it an embodiment-native world–action model.

3. Tokenizer design

LingBot-VA reuses the Wan 2.2 backbone's tokenizer. LingBot-VA 2.0 trains its tokenizer separately. The figure shows its semantic visual–action tokenizer, with parallel visual reconstruction, semantic alignment, and latent-action extraction branches that jointly form the tokenizer-training loss.

![LingBot-VA 2.0 semantic visual–action tokenizer architecture](../../../assets/images/06-embodied-ai-world-models/33-02/semantic-visual-action-tokenizer.png)

(1) Visual reconstruction: A reconstruction loss similar to that used in VAE training.

(2) Semantic alignment: Freeze a perception encoder as the teacher policy and align generated visual features with it, elevating the tokenizer's view of images from low-level pixels to high-level semantic concepts.

(3) Latent-action extraction: Train an inverse dynamics model (IDM) and forward dynamics model (FDM). The former extracts action features between two frames from their state features; the latter attempts to reconstruct the later frame's state features from the earlier frame and those action features. This loss ensures that tokenizer latents reflect action dynamics rather than merely compressing pixel features.

### (2) Training Data (LingBot-VA 2.0)

1. General images, text, and videos: Provide the broadest appearance and dynamics priors.

2. Robot data: Teleoperation data, AgiBot, OXE (including DROID), and others.

3. First-person human videos: 65.4k episodes covering more than 3,000 scene–task combinations, with every frame annotated with both hands' 6-DoF poses and 22 finger-joint angles per hand. All human hands are aligned to parallel grippers.

4. Synthetic data for subsequent ICL: A multimodal LLM first analyzes tasks and generates image-editing instructions, converting the first robot frame into an initial observation of human manipulation. A video-generation model then synthesizes human-manipulation videos. Finally, a VLM scores the robot data and generated human videos for task-semantic fidelity and physical plausibility; only qualified videos are paired with the original robot videos. This covers more than ten robot-video sources, over 5,000 tasks, and more than 50,000 human–robot pairs.

### (3) Pretraining Tasks (LingBot-VA 2.0)

T2I (text-to-image): Establish visual–semantic alignment and connect latent space with language.

T2V (text-to-video): Learn general temporal dynamics from Internet-scale videos.

TI2VA (text-image to video-action): Predict future latents and model latent actions between adjacent states through inverse dynamics.

ICL (in-context learning): Use human demonstration videos as conditioning context and require the model to generalize to manipulation trajectories for new instances without updating weights.

HCT (human–robot co-training): Jointly train human videos and robot data in a shared world model to extract general cross-embodiment action representations.

To better inherit prior knowledge, all five tasks are jointly optimized throughout training, changing only their mixture weights over time. Early training concentrates weight on T2I, shifts toward T2V in the middle, and focuses on TI2VA, ICL, and HCT later, while retaining nonzero regularization from T2l, T2V, and other tasks to avoid forgetting basic world knowledge.

### (4) Asynchronous Inference

Overall, the model uses “diffusion within chunks and autoregression between chunks.” A diffusion chunk can contain several tokens. Number the chunks 0, 1, 2, and so on.

To avoid pauses, the model must predict the next action in advance while the robot executes the current one, enabling asynchronous inference. LingBot-VA 2.0 uses the following workflow:

![LingBot-VA 2.0 asynchronous inference timeline](../../../assets/images/06-embodied-ai-world-models/33-02/asynchronous-inference-timeline.png)

Action 0 can begin execution only after it is predicted. While action 0 executes, the model must predict state 2 and action 1, with state 1 in its context still being a prediction. Continually accumulating predicted values in context would cause compounding errors. Therefore, after action 0 finishes and reaches state 1, and prediction of state 2 and action 1 is complete, the actual state 1 is written into context to replace the earlier prediction. Only state 2 then remains predicted. The system executes action 1 while predicting state 3 and action 2, and repeats, ensuring that context contains only one predicted state and model prediction can occur concurrently with action execution.

## II. Unified Autoregressive Prediction of Physical Tokens

In embodied AI, future visual states and actions are fundamentally high-dimensional continuous signals. Papers such as PAR and PhysGen, as well as NVIDIA's DreamDojo, propose encoding states and policies together as “physical tokens,” generating them autoregressively in a unified way, and decoding them downstream into state and policy outputs.

### (1) Core Idea

Encode past videos and actions as unified physical tokens. Let the set of video-frame tokens at time $t$ be $o_t$ (360 tokens per $o_t$ in PhysGen), and the action-token set be $a_t$ (eight tokens per $a_t$ in PhysGen). The context is $[\mathrm{prompt},o_1,a_1,o_2,a_2,\ldots,o_t,a_t]$, jointly predicting the next $[o_{t+1},a_{t+1}]$. Predicted $[o_{t+1},a_{t+1}]$ is then diffusion-decoded into continuous future states (video) and actions to execute.

More specifically, $o_t$ and $a_t$ differ: tokens within $o_t$ can be generated in parallel, while those within $a_t$ are strictly sequential in time. Multi-token prediction extends the microscopic planning horizon within an action chunk. Anticipating even two or three very small arm-pose steps makes autoregressive action trajectories more coherent, effectively avoiding control “shortsightedness” and jitter.

![Autoregressive architecture with unified physical tokens](../../../assets/images/06-embodied-ai-world-models/33-02/unified-physical-token-model.png)

### (2) Autoregressive Transformer

The Transformer backbone uses autoregressive generation with the following causal masks:

1. Within a frame: Block-level full attention

For the video-frame portion of physical tokens, the mask allows all image patches in the same frame to “see” one another. Spatial information within one image exists simultaneously, without temporal ordering. Full attention ensures complete extraction of the frame's global spatial features.

2. Within actions: Temporal causal attention

For the action portion of physical tokens, strict temporal causal masking is applied. Even within an action chunk at one time step, tokens are ordered, and earlier action tokens cannot attend to later ones. Actions form a continuous sequence occurring linearly in time and must follow the physical principle that “the past determines the future, and the future cannot interfere with the past.”

3. Actions attend unidirectionally to vision

Within the same time step t, action tokens may attend unidirectionally to video tokens. This teaches the model inverse-kinematic reasoning: “given a target state, infer the required action.”

4. Across time steps: Global temporal causality

Strict temporal causal masking across different time steps predicts time t+1 from information before time t, preventing future-frame information from leaking into past planning.

### (3) Diffusion Decoder

Unlike conventional discrete classification, PhysGen introduces a DiT-based denoising process that uses diffusion loss to estimate probability distributions directly in continuous space.

During training, the model predicts action noise to minimize mean squared error:

$$
\mathcal{L}(P_n,Z_n)=\mathbb{E}_{\varepsilon,t}\left[\left\|\varepsilon-\varepsilon_\theta(P_{n,t}\mid t,Z_n)\right\|^2\right].
$$

The inference-time denoising formula is:

$$
\begin{aligned}
P_{n,t-1}
&=\frac{1}{\sqrt{\alpha_t}}
\left(
P_{n,t}-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}
\varepsilon_\theta(P_{n,t}\mid t,Z_n)
\right)
+\sigma_t\varepsilon.
\end{aligned}
$$

Denoised physical tokens are finally separated and decoded into predicted video frames and robot actions to execute in the physical world.

### (4) BOA Token

Causality in the real physical world is strict. Suppose a robot must complete a task:

- Initial moment $t=0$: The robot opens its eyes and sees a cup on the table, obtaining the initial visual observation (Frame 0). In this brief instant, however, it has not yet taken any action.
- First step $t=1$: The model processes Frame 0, decides to extend the arm, and outputs the first action (Action 1). The action changes the physical world, after which the robot sees a new image (Frame 1).

PhysGen concatenates frame and action tokens into a physical token $P_n$, encoded as:

$$
P_n=[O_n;E_{A,n}].
$$

Following this physical intuition, the history input consists of two parts:

- Visual-observation sequence: $\lbrace O_0,O_1,O_2,\ldots,O_{N-1}\rbrace$, starting at 0 and containing $N$ initial frames.
- Corresponding action sequence: $\lbrace A_1,A_2,\ldots,A_{N-1}\rbrace$, starting at 1 and containing $N-1$ initial actions.

Forcing them into $P_n$ creates an awkward temporal offset: $P_0=[O_0;\text{nothing}]$, while $P_1=[O_1;A_1]$ and $P_2=[O_2;A_2]$.

To perfectly align these unequal-length sequences mathematically, the authors prepend a learnable placeholder to the action sequence, called the begin-of-action (BOA) token. The initial physical token becomes:

$$
P_0=[O_0;\mathrm{BOA}],
$$

Every subsequent physical token thus consistently contains one visual state and its corresponding action token.

### (5) Training Workflow

1. Pretraining

During pretraining, the backbone is trained only on vast video data. It sees no robot action commands at this stage and learns “how the world works,” such as cups breaking when dropped or objects moving when touched, representing purely physical and temporal dynamics.

This stage trains only the language tokenizer, visual tokenizer (3D-VAE), autoregressive Transformer backbone, and frame diffusion generator; action modules do not participate.

2. Action fine-tuning

To enable robot control, the authors fine-tune on specific downstream tasks with action-labeled demonstrations. Because the model already understands physical-world operation from the first stage, it requires very little action data.

An action tokenizer (an MLP) and lightweight action decoder (Action-DiT) are added. During fine-tuning, the model learns to alternate “360 video-frame tokens + eight action tokens” after seeing BOA, and the system routes tokens to different decoders alternately according to their counts.

## References

- Li, L., Zhang, Q., Luo, Y., et al. (2026). [Causal World Modeling for Robot Control](https://arxiv.org/abs/2601.21998). arXiv:2601.21998. (LingBot-VA)
- Zhang, Q., Li, L., Zhang, L., et al. (2026). [Native Video-Action Pretraining for Generalizable Robot Control](https://arxiv.org/abs/2607.08639). arXiv:2607.08639. (LingBot-VA 2.0)
- Song, Z., Qin, S., Chen, T., Lin, L., & Wang, G. (2025). [Physical Autoregressive Model for Robotic Manipulation without Action Pretraining](https://arxiv.org/abs/2508.09822). arXiv:2508.09822. (PAR)
- Song, Z., Li, Q., Qin, S., et al. (2026). [Learning Physics from Pretrained Video Models: A Multimodal Continuous and Sequential World Interaction Models for Robotic Manipulation](https://arxiv.org/abs/2603.00110). arXiv:2603.00110. (PhysGen)
- Gao, S., Liang, W., Zheng, K., et al. (2026). [DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos](https://arxiv.org/abs/2602.06949). arXiv:2602.06949. (NVIDIA DreamDojo)
