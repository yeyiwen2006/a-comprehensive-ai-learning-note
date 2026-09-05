---
title: "33.3 World–Action Models without Video Generation at Inference"
chapter_title: "World–Action Models"
section_id: "33-03"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/33.世界-动作模型/33.3 推理时不生成视频的世界-动作模型.docx"
status: "manually rebuilt and checked against Word"
ocr: "all Word-visible text and formula images manually transcribed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 33.3 World–Action Models without Video Generation at Inference

## I. Background

Although world–action models (WAMs) help address VLA limitations, generating future video frames introduces substantial inference latency. The physical world does not pause, so robots require low-latency real-time control, and excessive delays are often unacceptable. Enabling WAMs to operate without generating complete videos at inference has therefore become an important research direction.

## II. Fast-WAM: Discarding the Video Branch at Inference and Using Only the Action Branch

### (1) Model Architecture and Training Workflow

Fast-WAM uses Wan2.2 as its base, reusing the video DiT, text encoder, and video VAE and converting them into an MoT architecture through post-training.

**Step One: Precise Token Grouping**

Before entering MoT, all input information is strictly divided into three token groups:

1. Visual anchor $f_0$: Clean, unnoised latent features of the current first-frame observation.
2. Future video $f_1,\ldots,f_h$: Noisy latent features of future frames, present only during training.
3. Action sequence $a_1,\ldots,a_h$: Noisy features of the action chunk to generate.

“First frame” here refers to the anchor relative to the “future video chunk” originally to be generated. The model discards future-image generation and feeds only this single current image into the video backbone for one unidirectional forward pass, extracting a world representation and passing it directly to the action expert to generate subsequent actions. This gives Fast-WAM extremely fast real-time responses similar to standard VLAs.

**Step Two: Global Text Cross-Attention Injection**

As the three token groups enter each DiT layer, all attend to language-feature vectors through cross-attention. Mathematically, $f_0$, $f_{1:h}$, and $a_{1:h}$ serve as queries, while text features serve as keys and values. Thus, current-environment visual extraction, future physical prediction, and specific robot-action planning all remain guided by human language instructions.

**Step Three: Structured Self-Attention Masks**

After text information is injected, tokens interact through self-attention inside the DiT. Strict attention masks “combine” and “isolate” the Video DiT and Action DiT:

- Internal Video DiT interaction: Noisy future-video tokens $f_1,\ldots,f_h$ attend bidirectionally within the video branch and may attend to clean first-frame features $f_0$, allowing future prediction from the initial state.
- Internal Action DiT interaction: Action tokens $a_1,\ldots,a_h$ attend bidirectionally within the action branch and may attend to clean first-frame features $f_0$, allowing coherent action planning from the current visual state.
- Uncrossable boundary: Action tokens must never attend to future-video tokens. Meanwhile, clean first-frame tokens $f_0$ do not attend to any other tokens, avoiding noise contamination.

The fundamental reason for strict separation is that, if action features see future-video features during training, the action expert infers actions from “results that have already happened in the future,” developing a dependence. For extremely low deployment latency, however, Fast-WAM removes the entire future-video-generation branch. If the model learned to depend on future videos, their absence at inference would cause the action branch to collapse from missing information. Structured masks force action prediction to rely only on the present—the first frame and text—fundamentally decoupling training-time video assistance from inference-time action generation.

### (2) Inference Workflow

Explicit video generation is completely discarded at inference:

1. Single forward pass: Retain only clean first-frame latent features and pass them once, unidirectionally, through the video backbone to extract a physically meaningful latent world representation $z(o,l)$.
2. Direct action denoising: The action-expert DiT directly parameterizes the action distribution from this representation:

$$
p_\theta(a_{1:H}\mid o,l)=p_\theta(a_{1:H}\mid z(o,l)).
$$

3. Output and execution: Without instantiating and denoising future frames, action sequences require only ten denoising steps with classifier-free guidance set to CFG=1.0.

### (3) Experimental Results

- Fast-WAM performs very similarly to variants that explicitly generate futures (Joint and IDM).
- Removing the video co-training task drops RoboTwin success to 83.8% and real-robot towel-folding success sharply to 10%.

The key to stronger WAM capabilities is therefore that video prediction during training reshapes representations of the physical world, not that future images are explicitly generated at inference. Fast-WAM absorbs world-model physical priors during training while achieving VLA-like real-time responses in deployment.

## III. Dyna-2: The Video Branch Provides Visual–Text Features to the Action Branch without Prediction at Inference

Dyna-2 is a WAM trained on large quantities of data centered on first-person human videos, also using joint video- and action-generation objectives. Scaling first-person human video data is discussed separately in the embodied-data chapter; the focus here is Dyna-2's architecture.

### (1) Model Architecture

1. Video Transformer layers

Inputs: Observed frames Vc={v\_1,v\_2,...,v\_N}; noisy future frames z\_t={z\_N+1,...,z\_N+H}; text instruction T.

Attention: Frames use causal attention. Within Vc, v\_i can attend only to v\_1,...,v\_i-1; z\_t attends unidirectionally to Vc. Frames cross-attend to text, using video as queries and text as keys and values, so video vectors absorb textual information.

Outputs: Video Transformer layers turn Vc into observed visual features Vc’ that absorb text information but not z\_t information (a function of Vc and T). Features after one to three Video Transformer layers are injected into the Action Transformer for action generation; research shows that useful visual features emerge after few layers without waiting for deeper processing. The layers turn z\_t into denoised future frames z\_t’ (a function of Vc, z\_t, and T), using flow matching to compute the visual loss during training.

2. Action Transformer layers

Inputs: Noisy action chunk a\_t={a\_N+1,...,a\_N+H}; robot proprioceptive quantities (such as current joint positions, joint velocities, gripper state, and end-effector state); observed visual features Vc’ from Video Transformer layers.

Attention: The action chunk a\_t uses internal bidirectional attention and cross-attends to observed visual features Vc’, with actions as queries and visual features as keys and values. It does not directly attend to text but receives textual information indirectly through Vc’. It obtains no information from future visual predictions z\_t’.

Output: Denoised action chunk a\_t’ is both the final inference result and the basis for the action loss during training.

The Action Transformer has fewer layers. Inference only needs these layers and a few shallow Video Transformer layers to generate actions, ensuring high speed.

3. Example

Suppose the robot has observed three frames $I_1,I_2,I_3$, and training data also contains true future frames $I_4,I_5,I_6$. Future frames are encoded into latents and noised to form $z_{4,t},z_{5,t},z_{6,t}$. The Video Transformer input can be illustrated as:

    past context                  noisy future
    I1  I2  I3       |        z4,t  z5,t  z6,t
            │
    Video Transformer (causal attention)
            │
    text ──cross-attention
            │
    predicted video velocity

The Action Transformer reads the world-model backbone's representations of observed videos and outputs actions; it does not read $z_{4,t},z_{5,t},z_{6,t}$ from the future-prediction branch. Training-time future video only shapes world-model representations and is not a required input for action inference.

## IV. Being-H0.7: Aligning “Action Prediction” with “Action Prediction Based on Future States” during Training

### (1) Core Idea

Being-H0.7 does not need pixel-level future prediction. Instead, it directly aligns action prediction with future-state-based action prediction during training. It introduces learnable latent queries that progressively extract task-relevant information from multimodal context through Transformer layers, compressing and reorganizing it to generate actions. During training, observed future states are also compressed into latent space and extract task-relevant information from multimodal context, compressing and reorganizing it to generate actions. The former, the prior branch, is aligned with the latter, the posterior branch.

- Prior branch: The main branch used in real deployment, relying only on current instructions and observations to generate actions through latent queries.
- Posterior branch: A training-only auxiliary branch that replaces the main branch's latent queries with future-observation embeddings extracted by a vision encoder.
- Alignment: Compute hidden-feature alignment loss between the branches, forcing the prior to infer the same feature representations as “seeing the future” while observing only the present.

### (2) Training Workflow

**Step 1: Context Encoding and Sequence Construction**

Given task instruction $x$, historical observations $o_{-H:0}$ spanning $H$ time units, and robot proprioceptive state $s$, map and concatenate these features. Before the action-chunk prediction sequence $a_{0:T}$, insert a latent-query matrix $Q\in\mathbb{R}^{K\times d}$ with $K$ queries of dimension $d$, constructing the full prior-branch Transformer sequence:

$$
S=[x;o_{-H:0};s;Q;a_{0:T}].
$$

**Step 2: Future-Feature Extraction for the Posterior Branch**

During training, also obtain true future observations $\tilde{o}_{0:T}$. A frozen pretrained ViT and Perceiver resampler $E$ compress them into future embeddings of the same size as $Q$:

$$
z^{\mathrm{post}}=E(\tilde{o}_{0:T})\in\mathbb{R}^{K\times d}.
$$

**Step 3: Dual-Branch Forward Pass with a Mixture of Transformers**

For computational efficiency, instead of two large serial forward passes, a mixture-of-Transformers architecture merges the branches into a long sequence with a dual-branch attention mask: both see shared context $c=[x;o_{-H:0};s]$, but their tokens cannot see each other.

**Step 4: Loss Computation**

The framework is jointly optimized with three groups of losses.

1. Action flow-matching loss: Under the diffusion-generation paradigm, compute each branch's denoising velocity-field loss for noisy actions $\tilde a_t$. If the true action is $a$, noising time is $t$, and target velocity is $u_t=a-\tilde a_t$, then:

$$
\mathcal{L}_{\mathrm{FM}}^{\mathrm{prior}}=\left\|v_\theta^{\mathrm{prior}}(a_t,c)-u_t\right\|_2^2,
$$

$$
\mathcal{L}_{\mathrm{FM}}^{\mathrm{post}}=\left\|v_\theta^{\mathrm{post}}(a_t,c,z^{\mathrm{post}})-u_t\right\|_2^2.
$$

2. Joint hidden-state alignment loss: At aligned Transformer layer $l$, extract prior hidden state $h_l^{\mathrm{prior}}$ and posterior hidden state $h_l^{\mathrm{post}}$ and compute MSE:

$$
\mathcal{L}_{\mathrm{align}}=\frac{1}{L}\sum_{l=1}^{L}\left\|h_l^{\mathrm{prior}}-h_l^{\mathrm{post}}\right\|_2^2.
$$

3. Anti-collapse regularization: Penalize norm and rank to prevent alignment from driving hidden features toward all zeros or a single direction. Norm regularization is:

$$
\mathcal{R}_{\mathrm{norm}}(h)=\left[\operatorname{ReLU}\!\left(\tau-\|h\|_2\right)\right]^2.
$$

   Rank regularization is expressed through the entropy of normalized eigenvalues $p_i$ of Gram matrix $G$:

$$
\mathcal{R}_{\mathrm{rank}}(H)=\sum_{i=1}^{B}p_i\log p_i.
$$

### (3) Inference Workflow

Decoupling feature alignment from forward generation makes inference lightweight:

1. Observation collection: Robot cameras capture the current scene in real time, processing and concatenating it into a context vector.
2. Pure prior-branch execution: Discard the large posterior branch and all future-video inputs, retaining only the Transformer backbone, latent queries $Q$, and flow-matching generation head for diffusion.
3. Action-chunk output and UAC asynchronous control: Generate a continuous future action chunk $a_{0:T}$ at once. Real deployment combines this with universal asynchronous chunking (UAC): the server asynchronously infers future actions, while the robot driver continuously takes history-prefix-locked actions from a thread-safe buffer for execution. Decoupling them enables smooth dynamic control with approximately three to four milliseconds of latency per step.

## References

- Yuan, T., Dong, Z., Liu, Y., & Zhao, H. (2026). [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666). arXiv:2603.16666.
- Dyna Robotics. (2026). [Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models](https://www.dyna.co/dyna-2). Official research report.
- Luo, H., Zhang, W., Feng, Y., et al. (2026). [Being-H0.7: A Latent World-Action Model from Egocentric Videos](https://arxiv.org/abs/2605.00078). arXiv:2605.00078.
