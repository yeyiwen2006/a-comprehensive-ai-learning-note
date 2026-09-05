---
title: "31.1 VLA Model Architectures and Training Methods"
chapter_title: "VLA Models"
section_id: "31-01"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/31.VLA模型/31.1 VLA模型的架构与训练方法.docx"
status: "manually rebuilt and checked against Word"
ocr: "all Word-visible text and formula images manually transcribed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 31.1 VLA Model Architectures and Training Methods

Vision–language–action (VLA) models are fine-tuned from pretrained vision–language models (VLMs) and represent an important paradigm for embodied AI in the era of large models. They break with the traditional robot's fragmented, modular “perception–planning–control” design, directly mapping semantics to actions in the physical world.

## I. Core Idea

The core idea of VLA models is deep, end-to-end multimodal integration. A pretrained vision–language model (VLM) serves as the “brain,” and robot action outputs are transformed into a special “language,” enabling the large model to directly command physical hardware.

### (1) Input and Output Modalities

1. Vision: A visual encoder (such as ViT or SigLIP) divides input images (such as an RGB camera stream) into patches and converts them into visual tokens, giving the robot the ability to understand spatial relationships and physical properties.

2. Language: The model receives human natural-language instructions and draws on the vast Internet knowledge accumulated by the pretrained large model to provide logical reasoning and “common sense.”

3. Action: VLA lexicalizes actions, discretizing continuous robot-control parameters (such as arm joint torques and end-effector coordinates) and incorporating them into the large language model's vocabulary as special tokens.

### (2) Basic Architecture

1. VLM: Usually obtained by fine-tuning a pretrained model, it handles reasoning and high-level decisions based on visual and language inputs.

2. Action head: Usually also a Transformer, it is combined and jointly trained with the VLM, turning a standalone VLM into a VLA that can output robot actions. The VLM's output is fed into the action head as context.

## II. The VLA Paradigm with an Autoregressive Action Head

### (1) Basic Principles

Suppose that at time step $t$, the agent receives a visual-observation sequence $o_{\le t}\in\mathcal{O}$ and a natural-language instruction $l\in\mathcal{L}$. The model aims to output the optimal action $a_t\in\mathcal{A}$.

In the autoregressive VLA paradigm represented by RT-2 (Robotic Transformer 2) and OpenVLA, a continuous action $a_t$ is discretized into a series of action tokens $(z_1,z_2,\ldots,z_K)$. For example, a 7-degree-of-freedom (7-DoF) robotic-arm action can be represented as spatial translation, rotation, and gripper opening.

The training objective is to maximize the log-likelihood of action tokens conditioned on observations and instructions:

$$
\begin{aligned}
\mathcal{J}(\theta)
&=
\mathbb{E}_{(o,l,a)\sim\mathcal{D}}
\left[
\sum_{k=1}^{K}
\log P_{\theta}(z_k\mid z_{<k}, o_{\le t}, l)
\right]
\end{aligned}
$$

Here, $\theta$ denotes the neural-network parameters and $\mathcal{D}$ is a mixed dataset containing human-expert demonstrations and Internet image–text data. This fully and equivalently transforms action prediction into a next-token-prediction task, similar to ChatGPT generating text.

### (2) Workflow

1. Multimodal input processing: Step 1.1, obtain the current camera RGB image $o_t$, extract features with a vision encoder, and align them to the language model's feature space through a projector to generate a visual-token sequence; step 1.2, receive the user instruction $l$ and use a text tokenizer to generate a language-token sequence.

2. Context concatenation and cross-modal attention: Step 2.1, concatenate the system prompt, visual tokens, and language tokens into `[System Prompt, Image Tokens, Language Tokens]`; step 2.2, feed this sequence into the Transformer decoder backbone, using self-attention to deeply ground instruction semantics in the image's physical features.

3. Autoregressive action generation: Step 3.1, the Transformer uses the context to predict the current step's action tokens one by one. For example, it first predicts the token representing $A_x$, then uses $A_x$ as known context to predict $A_y$, and continues until it outputs the gripper-action token.

4. Detokenization and physical control: Step 4.1, map the generated action-token sequence $(z_1,\ldots,z_K)$ back into continuous physical values to produce actual control commands; step 4.2, send them to the robot's low-level controller (such as a PID controller) to drive the motors.

### (3) Advantages and Disadvantages

Autoregressive VLAs directly reuse LLM architectures and naturally inherit chain-of-thought and commonsense reasoning from Internet-scale data. Their generalization capability is extremely strong, but autoregressive generation speed limits their ability to perform high-frequency continuous control.

## III. The VLA Paradigm with a Diffusion/Flow-Matching Action Head

The core motivation of this paradigm is to eliminate discretization errors introduced by “action lexicalization.” Rather than forcing actions into tokens in a language vocabulary, it uses generative continuous-time models (diffusion models or flow matching) to directly generate high-dimensional, high-frequency action-trajectory chunks in continuous space. Representative models include Physical Intelligence's $\pi_0$ (Pi-Zero) and ForceVLA.

Diffusion/flow-matching VLAs typically adopt a decoupled-and-recombined **“VLM + action expert”** architecture:

1. Multimodal conditioning encoder: Inherits a pretrained vision–language model (such as PaliGemma, or SigLIP combined with an LLM backbone) and processes image sequences $o$ and language instructions $l$ into deep contextual conditioning embeddings.

2. Action expert for continuous generation: Typically a diffusion Transformer (DiT) or U-Net, it receives these conditioning embeddings and iteratively denoises pure-noise signals through cross-attention or concatenated self-attention, ultimately outputting a continuous physical-action chunk $A_t=(a_t,a_{t+1},\ldots,a_{t+H-1})$.

Take the currently advanced optimal transport flow matching approach (similar to the design in $\pi_0$) as an example. Suppose the base noise distribution is $x_0\sim p_0=\mathcal{N}(0,I)$, the true action-trajectory distribution is $x_1\sim p_1$, and the multimodal condition is $c=\mathrm{Encoder}(o,l)$. The model constructs a straight probability path connecting noise to true actions:

$$
x_t=(1-t)x_0+t x_1
$$

The action generator $v_\theta$ aims to fit the true velocity field, with the objective:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{FM}}(\theta)
&=
\mathbb{E}_{\substack{t\sim \mathcal{U}(0,1)\\ x_0\sim p_0,\ x_1\sim p_1}}
\left[
\left\|v_\theta(x_t,t,c)-(x_1-x_0)\right\|^2
\right]
\end{aligned}
$$

Here, $(x_1-x_0)$ is the constant gradient of the true path. Compared with traditional DDPM diffusion models, flow matching has smoother, straighter ODE trajectories and can generate high-fidelity continuous action signals in very few inference steps (even just a few).
Workflow:

1. Feature extraction: The VLM receives current camera observations and language $l$, then performs a forward pass to generate the multimodal conditioning vector $c$.

2. Noise initialization: Sample a pure-noise trajectory of length $H$ from a standard normal distribution, $x_0\sim\mathcal{N}(0,I)$.

3. ODE solving: Across time steps $t\in[0,1]$, the action generator $v_\theta(x_t,t,c)$ uses context $c$ to compute the gradient at the current state and update it progressively through a numerical integrator (such as Euler's or Heun's method).

4. Action execution: Take the ODE endpoint as the predicted action chunk. The system uses model predictive control (MPC/receding-horizon control), executing only the first few actions before acquiring new visual observations and beginning the next closed-loop inference round.
Advantages and disadvantages: Action outputs are no longer discrete tokens. Instead, diffusion models or flow matching directly generate continuous action trajectories, providing significant advantages for tasks requiring extremely high precision and dexterity (such as clothes folding and dexterous-hand manipulation), but potentially underperforming autoregression in maintaining causal relationships over long-horizon tasks.

## IV. Unified Representations and the Latent-Action Paradigm

Robot hardware embodiments differ from one another (and from human first-person perspectives). To keep VLA models from being limited to specific hardware and allow scaling with vast quantities of videos without action labels, we introduce unified latent actions that are independent of particular hardware, such as “grasp the object at the upper right,” rather than specific mechanical operations such as “raise the first joint of the right arm by 30 centimeters.”

Within the classification of embodied AI architectures, this lies between hierarchical and end-to-end approaches: the information passed between models is differentiable, and gradients can (in some designs) propagate directly from lower to higher levels, or alignment can be completed within the latent space of the same world model.

### (1) Core Architecture

The core architecture can still be divided into two components: a multimodal conditioning encoder inherits a pretrained VLM and encodes image sequences $o$ and language instructions $l$ into contextual conditioning embeddings; a continuous action generator receives these embeddings and generates unified latent-action chunks $A_t=(a_t,a_{t+1},\ldots,a_{t+H-1})$. The difference is that high-level outputs are not necessarily low-level control quantities specific to a robot, but latent-action representations shared across embodiments.

### (2) Training Stage One: Unsupervised Learning of a Latent-Action Space

This stage requires no low-level physical-control signals (such as joint torques); the model learns solely by watching vast video sequences. It typically introduces an inverse dynamics model based on a vector-quantized variational autoencoder (VQ-VAE): given two consecutive video frames $(o_t,o_{t+1})$, the encoder extracts changes in visual and physical states and generates continuous hidden features; a vector-quantization module maps these features into a discrete codebook, extracting task-centric latent actions $z_t$; a forward dynamics model receives the current frame $o_t$ and latent action $z_t$ and attempts to reconstruct the next frame $\hat{o}_{t+1}$.

The loss consists of reconstruction and codebook-regularization terms:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{VQ}}
&= \|o_{t+1}-\hat{o}_{t+1}\|^2
+\|\mathrm{sg}[\mathrm{Encoder}(o_t,o_{t+1})]-e_{z_t}\|^2
\\
&\quad+
\beta\|\mathrm{Encoder}(o_t,o_{t+1})-\mathrm{sg}[e_{z_t}]\|^2
\end{aligned}
$$

Here, $e_{z_t}$ is an embedding vector in the codebook, and $\mathrm{sg}[\cdot]$ is the stop-gradient operation.

### (3) Training Stage Two: Joint Policy Optimization and Embodiment Alignment

After a cross-embodiment latent-action space is obtained, formal training of the large VLA model begins to bind semantics to physical hardware. First, the VQ-VAE encoder trained in Stage One converts robot videos with language instructions into discrete “instruction–image–latent action $z_t$” sequence data. Next, the VLM backbone is trained with standard next-token prediction to output the correct latent-action distribution $P_\theta(z_t\mid o_{\le t},l)$ given image and text context. Finally, a lightweight embodiment-aware decoder or joint diffusion module is introduced. It concatenates the VLM's latent-action output $z_t$ with the current robot's specific embodiment identifier and is trained with a small amount of data containing true low-level action labels:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{Action}}
&=
\mathbb{E}\left[
-\log P_\phi(a_t\mid z_t,o_t,\mathrm{embodiment\_id})
\right]
\end{aligned}
$$

Some current industry implementations use autoregression at the high level to fit a mature ecosystem and diffusion at the low level to reduce latency.

## V. Limitations of VLA

Because VLA models are fundamentally fine-tuned from language models, they still use natural-language space as the “brain's” core representation layer, with actions serving as an auxiliary output modality. This design gives VLA clear advantages in semantic understanding and task generalization, but also means that continuous physical-world states, dynamic constraints, and action consequences are not directly modeled.

VLAs also typically predict actions directly from current observations and language instructions, making errors accumulate over long-horizon tasks. When a robot enters states poorly covered by training data, the model lacks explicit future prediction and self-correction mechanisms. Introducing world models, reinforcement learning, or more unified physical representations is therefore an important direction for improving VLA reliability and long-term planning.

## References

- Brohan, A., Brown, N., Carbajal, J., et al. (2023). [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818). arXiv:2307.15818.
- Kim, M. J., Pertsch, K., Karamcheti, S., et al. (2024). [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246). arXiv:2406.09246.
- Black, K., Brown, N., Driess, D., et al. (2024). [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164). arXiv:2410.24164.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR.
- Ye, S., Jang, J., Jeon, B., et al. (2024). [Latent Action Pretraining from Videos](https://arxiv.org/abs/2410.11758). arXiv:2410.11758.
- Bu, Q., Yang, Y., Cai, J., et al. (2025). [UniVLA: Learning to Act Anywhere with Task-centric Latent Actions](https://arxiv.org/abs/2505.06111). arXiv:2505.06111.
- Chen, Y., Ge, Y., Tang, W., et al. (2024). [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](https://arxiv.org/abs/2412.04445). arXiv:2412.04445.
