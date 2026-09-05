---
title: "29.2 Backbone Architectures and Training for Unified Multimodal Understanding and Generation"
chapter_title: "Unified Multimodal Understanding and Generation Models"
section_id: "29-02"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/29.统一多模态理解-生成模型/29.2 多模态统一理解-生成模型的主干架构与训练.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 29.2 Backbone Architectures and Training for Unified Multimodal Understanding and Generation

## I. Output Paradigms of the Backbone Network

### (1) Autoregressive Text and Diffusion Images with a Shared Transformer (ByteDance BAGEL, Meta Transfusion, and SenseTime SenseNova-U1)

Using the same Transformer, the model changes its masks when outputting different modalities, switching directly between two states:

- **Text output**: It follows the next-token-prediction (autoregressive) paradigm of large language models and outputs text word by word.
- **Visual output**: This is no longer autoregressive. Instead, a denoising process similar to a diffusion model or rectified flow generates a distribution in a continuous latent space, which is finally rendered into a high-fidelity pixel image by a dedicated image decoder (such as a VAE decoder).

Causal attention is used when outputting text. When outputting images, attention is causal with respect to the preceding context and bidirectional within the image. Tokens are processed in parallel for visual output. The model predicts the flow velocity and uses it to calculate the position $z$ after that denoising step.

### (2) Purely Autoregressive Generation (LongCat-Next and DeepSeek Janus)

DeepSeek Janus and LongCat-Next adopt purely autoregressive generation and causal attention. Both text and image tokens are discretized into vocabularies (different modalities have their own vocabularies). In the same Transformer, every generation step is autoregressive: without first distinguishing modalities, it predicts a discrete next-token vector (next-token ID) from the context, then sends the generated vector to the detokenizer for its modality to convert the token back into language, images, or other forms. Special control tokens (such as `<image_start>`, `<image_end>`, and `<audio_start>`) delimit the contextual boundaries between modalities in the sequence.

In LongCat-Next, an image patch consists of eight levels of tokens (using residual vector quantization, from the overall structure to details). Each step predicts one image patch and only predicts its first-level token. Once the token sequence of an entire image has been generated, a self-attention layer first predicts the second- through eighth-level tokens in parallel from the first-level tokens (the self-attention layer takes the first-level tokens plus seven learnable embedding vectors as input). A dual-track diffusion detokenizer then reconstructs a continuous pixel image:

1. **Track One: Structural Pixel Decoder**

   - **Operation**: It receives discrete visual tokens $Z$ and uses efficient convolutional and Transformer deconvolution networks to first reconstruct the image skeleton, object boundaries, and text layout $I_{\mathrm{struct}}$.
   - **Formula**: $I_{\mathrm{struct}}=D_{\mathrm{struct}}(Z)$.

2. **Track Two: Diffusion Pixel Refiner**

   - **Operation**: Taking the first track's output $I_{\mathrm{struct}}$ as a strong conditioning constraint, it uses a lightweight diffusion model for iterative denoising, injecting realistic lighting, materials, and ultrahigh-frequency texture details.
   - **Formula**: At step $t$ of reverse denoising, the refiner predicts textures based on the layout:

$$
x_{t-1}=\mu_{\theta}(x_t,I_{\mathrm{struct}},t)+\Sigma_{\theta}(x_t,t)\epsilon
$$

The final output is the high-fidelity image $x_0$.

### (3) DiDA: Train a Purely Autoregressive Model First, Then Distill and Fine-Tune Image Generation with a Diffusion Loss (Emu3.5)

During pretraining, SFT, and RL, next-token prediction is performed using a unified decoder-only autoregressive training method:

After this training is complete, the masking strategy is adjusted to change image generation from autoregression to few-step diffusion, improving both speed and efficiency. A single round of self-distillation is sufficient for the model to adapt rapidly to the new strategy:

When the model continuously outputs an entire segment of visual tokens in the sequence based on its current textual intent and generates an end-of-modality token, the underlying visual decoder (VAE decoder) **intervenes**. Its job is to “translate” this sequence of discrete visual codes back into a continuous RGB pixel grid, physically rendering the intermediate image state $V_t$ (for example, an image with bounding boxes drawn or puzzle pieces assembled).

- **Step A (serial text generation)**: When the system identifies that text needs to be generated, the engine invokes the standard autoregressive mode and emits text one token at a time.
- **Step B (parallel image generation)**: When an image-generation instruction is encountered, the FSM scheduler immediately switches states. The system allocates enough “noise tokens” for the entire image at once (for example, approximately 4,000 tokens for a 1024x1024 image).
- **Step C (discrete denoising)**: Using the bidirectional attention enabled by DiDA, the model refines these several thousand noise tokens in parallel through a small number of discrete denoising steps, recovering the final image tokens.
- **Step D (decoder output)**: Finally, the visual tokenizer's decoder (vanilla or diffusion-based) reconstructs these discrete tokens into a pixel image visible to humans.

### (4) Vision-First Discrete Diffusion (Muddit)

- **Unified discrete diffusion architecture**: It handles bidirectional image–text generation under the same Transformer architecture and the same decoding paradigm.
- **Strong visual prior**: Unlike the conventional approach based on a language prior, Muddit adopts a “vision-first” strategy. Its multimodal diffusion Transformer (MM-DiT) backbone is initialized with Meissonic, a thoroughly trained high-resolution text-to-image discrete diffusion model.
- **Lightweight text decoder**: Only an extremely lightweight linear head is added on top of the shared MM-DiT backbone as a text decoder, mapping the shared latent space back to text tokens.

Advantages over ordinary diffusion:

- **Ordinary diffusion generative models (and hybrid unified models)**: Most models focus on only one modality (such as text-to-image generation). Even some models that attempt to unify images and text (such as Transfusion) often adopt a “glue architecture”: an autoregressive LLM handles text, while continuous diffusion handles images, leaving their representation spaces and generation paradigms separate.
- **Muddit (complete, genuine unification)**: Both text and images are quantized into discrete tokens of equal status. Thus, whether generating text or images, Muddit uses exactly the same mathematical formulas, the same diffusion Transformer (MM-DiT) backbone, and the same set of loss functions.
- **Training paradigm**: Many earlier multimodal large models were dominated by a “language prior” (beginning with fine-tuning an open-source LLM), whereas Muddit is dominated by a “visual prior.” Its backbone is inherited from Meissonic, a purely high-quality text-to-image model, giving it a very strong ability to perceive spatial structure.

## II. Mixture of Transformer Experts

In dense models, tasks such as visual understanding and generation can easily conflict. Analysis through IsoFLOP scaling laws also reveals that vision is more “data-hungry,” whereas language is more “parameter-hungry.” We can therefore use MoE to “split the traffic.”

### (1) Allowing Modalities to Separate Naturally without Predefinition (LongCat-Next and Others)

- **High granularity and sparsity**: Fine-grained expert partitioning (such as reducing expert dimensions and setting $G=16$) is particularly important for multimodality. MoE can substantially increase the total parameter count without increasing the computational burden (active compute remains constant), significantly reducing text perplexity and improving image-generation quality.
- **Natural modality allocation**: Without human intervention, the router naturally assigns most experts to the more parameter-hungry “text,” while visual tokens share particular experts with text. The model mainly processes modality-specific features independently in shallow layers and fuses modalities only in deeper layers.
- **Closing the scaling gap**: After MoE is introduced, the scaling exponent for language rises ($D_{\mathrm{opt}}\propto C^{0.59}$), approaching that of vision ($D_{\mathrm{opt}}\propto C^{0.64}$). This greatly narrows the gap between their requirements during scaling, paving the way for genuinely multimodal foundation models with trillions of parameters.

### (2) Hard Separation (ByteDance BAGEL and Others)

- **Decoupled expert FFNs**: After fully local attention is computed, tokens are hard-routed to two different expert networks according to the properties of their modalities:
  - **Multimodal understanding expert**: It specifically processes text tokens and semantic tokens from the understanding encoder, focusing on logical operations.
  - **Multimodal generation expert**: It specifically processes latent tokens for image generation, focusing on simulating spatial and physical laws.

## III. Positional Encoding

Many vision–language models (such as NEO) use one-dimensional positional encoding for text and three-dimensional positional encoding for vision. To avoid affecting language positional encoding (whose dimension is the same as $d_{\mathrm{model}}$), some models extend the spatial positional encoding in vision (height $H$ and width $W$) into additional dimensions, while keeping the sequence positional encoding at $d_{\mathrm{model}}$ dimensions. To avoid inconsistent dimensions between visual and text tokens, spatial positional encoding is extended only in $Q$ and $K$, with no spatial positional encoding added to $V$.

However, this extension means that if we fine-tune the model into a unified multimodal understanding–generation model, its dimensions during visual generation will differ from those of language tokens. To maximize unification, SenseNova-U1 puts all positional encoding within $d_{\mathrm{model}}$ dimensions, then assigns different dimension segments of $Q/K$ to different RoPE encodings for $T/H/W$. This also reduces the parameter sizes of the $Q/K$ projection matrices (both shrink to $d_{\mathrm{model}}^2$). The paper does, however, retain two RoPE frequency scales: 5,000,000 for the sequential temporal axis and 10,000 for image spatial axes. This is because text sequences can be very long, so the temporal axis uses a larger RoPE theta; image spatial coordinates have different ranges and semantics, so spatial axes use another frequency scale.

If the positional encoding method is changed, the attention-layer parameters generally need to be fine-tuned first (with other parameters frozen) to let the model adapt, before post-training is performed.

## IV. Training Methods and Data

### (1) Pretraining from Scratch

Emu3.5:

To let the model learn temporal continuity and causal relationships in the world, Emu3.5's pretraining relies heavily on **“interleaved vision–language data”** extracted from long Internet videos (a total of 790 years of video material). Its algorithmic workflow has two stages:

1. **Stage One: Large-Scale General Alignment**
   - **Data assembly**: Video keyframes and automatic speech recognition (ASR) transcripts are aligned by timestamp and interleaved into long sequences of up to 32,768 elements.
   - **Optimization objective**: The model optimizes a standard cross-entropy loss on 10 trillion (10T) tokens. To prevent the large volume of visual tokens from overwhelming the text loss, the visual-token loss weight is adjusted to 0.5. AdamW is used with $\beta_1=0.9$, $\beta_2=0.95$, and $\epsilon=1.0\times10^{-8}$.

2. **Stage Two: High-Quality Enhancement Training**
   - **Data refinement**: Training continues on 3 trillion (3T) high-quality tokens, introducing rich multimodal annotations such as semantic segmentation and captioning.
   - **Resolution increase**: The upper bound of dynamic image resolution rises from 512x512 in the first stage to 1024x1024, encouraging the model to capture finer-grained visual details.

BAGEL:

VLM image–text pairs, video data, webpage data, and other videos -> sample approximately four frames -> use a VLM to generate short text describing the changes between each pair of frames -> construct an interleaved image–text–image–text sequence. This differs fundamentally from Emu3.5's direct NTP on sequential video frames. Emu3.5's video data preserves full spatiotemporal continuity, whereas BAGEL's video data is restructured into a discrete multi-image interleaved format. Temporal relationships are conveyed mainly through textual descriptions between frames, rather than through a dense sequence of the frames themselves.

LongCat-Next:

First train the all-modality encoder (tokenizer) and all-modality decoder (detokenizer), then train the backbone with a unified discrete next-token-prediction loss in latent space (during training, ground-truth labels are mapped through the encoder and cross-entropy is computed against predictions in latent space; during inference, the context consists of embeddings corresponding to token IDs directly predicted by the backbone itself). Although the loss is unified, the modality being generated is known, and the tokenizer and detokenizer for each modality have already been trained. The backbone is therefore trained to generate abstract representations of that modality at the corresponding positions, allowing downstream decoding into content in that modality.

Note that the only components trained separately are the tokenizer, which “turns inputs into discrete tokens” (more precisely, only the VQ-VAE needs training for vision, because the subsequent residual matching steps are hard-coded), and the detokenizer, which “turns discrete tokens into outputs.” The vector represented by each vocabulary token (the embedding layer) is part of main-model training. One engineering trick to prevent codebook collapse is to repeatedly reset low-usage entries using the current batch data, ensuring that all vocabulary entries are fully utilized.

### (2) Generative Pretraining Starting from a Multimodal Understanding Model

Taking SenseNova-U1 as an example:

Step one: Freeze the understanding branch and train only the generation branch. First learn text-to-image generation, then gradually introduce higher resolutions, image editing, reasoning, and interleaved image–text data.

Step two: Train understanding and generation together. The approximate data mixture is 0.33 understanding, 0.37 text-to-image, 0.24 editing, and 0.06 interleaved generation. The loss weights are $\lambda_1=0.1,\lambda_2=1.0$, giving the generation loss greater weight.

### (3) Post-Training

1. SFT: Fine-tune on high-quality samples and unify instruction-following capabilities across multimodal tasks.

2. RL (taking Emu3.5 as an example)

- **Algorithmic framework**: Group relative policy optimization (GRPO) is used for end-to-end joint reinforcement learning.
- **Unified reward system**: A multidimensional reward function is constructed. For example, text-rendering tasks use OCR accuracy rewards, while image generation uses aesthetic scores and CLIP similarity rewards.
- **Normalization**: To balance heterogeneous reward distributions across tasks, all reward signals are uniformly normalized to $[1,10]$ before backpropagation, preventing “reward hacking” on a single task.

Some tasks:

- **Visual narrative**: Given context, the model can generate mixed multi-image and multi-text outputs with continuous storylines and highly consistent characters.
- **Visual guidance**: It can understand multistep processes (such as cooking and crafts) and generate illustrated, step-by-step instructions, demonstrating strong causal reasoning.
- **World exploration and embodied manipulation**: In robotic manipulation, it can decompose long-video tasks into multiple subtasks. Given $Sub_i=(I_i,O_{t,i-1:t},O_{t,i+1})$ (where $I_i$ is the language instruction and $O$ is the observation sequence), Emu3.5 can accurately predict the evolution of physical laws, such as robotic-arm grasping and deformations during clothes folding, demonstrating its potential as a “world model.”

3. DMD distillation (diffusion models)

For slower visual-generation steps, distillation methods such as DMD can reduce the number of diffusion steps needed for image generation and improve inference speed.

### (4) DiDA Adaptation (Specific to Emu)

First, note that the output steps of an autoregressive model based on next-token prediction are not a high-dimensional vector, but the probabilities of logits in the vocabulary. Methods such as DMD therefore cannot be used to make the student learn the teacher's denoising trajectory. Meanwhile, the teacher is autoregressive and the student uses bidirectional diffusion, which also means that classical autoregressive-model distillation methods cannot be used.

Here, therefore, we perform only “data self-distillation”: the fully trained, maximally capable, purely AR version of Emu3.5 serves as the teacher and generates vast quantities of high-quality images in batches. Its own generated images are used as samples for DiDA adaptation. These images and their corresponding text instructions form the “self-distillation dataset” for training DiDA, at which point the teacher's task ends. Next, the system randomly samples noising time steps and has the student perform one denoising step (similar to diffusion-model training), but uses the autoregressive model's cross-entropy loss.

During inference, the denoising workflow becomes:

1. **Step 0 (a pure-noise starting point)**: Initialize all visual tokens to `[MASK]` (or special noise tokens) at once.
2. **Global prediction**: Through bidirectional attention, the model predicts token probabilities at all 4,096 positions at once.
3. **“Greedy elimination” and “selection of the best” (the key scheduling step)**:
   - The model does not directly adopt all 4,096 predictions.
   - It calculates confidence from the output probability distributions.
   - According to a predefined schedule, this step “commits” only a very small fraction of the highest-confidence tokens (for example, 10%). These tokens become “clean context” in subsequent steps.
   - The remaining 90% stay in the `[MASK]` state.
4. **Repeat**: Feed the 10% clean tokens and 90% noise tokens back into the model for the next prediction. Commit the next batch of highest-confidence tokens (for example, reaching 30% cumulatively).
5. **Finish**: After such iterations, all tokens are gradually committed, finally assembling a complete high-resolution image.

Why is DiDA adaptation performed after post-training?

DiDA training requires a specially constructed “natural-entropy dataset” containing image–text pairs and interleaved image–text sequences.

- The logical prerequisite of natural entropy is that a highly capable “teacher model” with very high output quality must first exist to provide alignment standards and generation trajectories.
- This “teacher model” is precisely the purely autoregressive Emu3.5 that has undergone full SFT and RL and reached peak generative capability. If these earlier steps were skipped and DiDA were trained directly, the model would be unable to produce high-quality data of its own to guide learning of parallel decoding.

In summary, the first three stages (pretraining, SFT, and RL) steadily build the “strongest brain” for understanding the multimodal world; the final DiDA stage is equivalent to fitting a separate parallel acceleration engine to the model's visual-output module before it leaves the factory. This decoupled design is the optimal solution for balancing maximum model capability and inference speed.

## V. GRPO Training Methods

### (1) UniGRPO

“UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation” is a study jointly proposed by the Chinese University of Hong Kong and ByteDance's Seed team. The paper primarily addresses the challenge of reinforcement-learning (RL) alignment in unified multimodal models.

The research team adopts a minimalist, highly scalable approach that rigorously formulates the entire reasoning-driven visual-generation process, “prompt -> thinking -> image,” as a unified Markov decision process (MDP). On this basis, it proposes UniGRPO, which uses sparse terminal rewards and group relative policy optimization (GRPO) to jointly optimize text-generation (autoregressive, AR) and image-generation (flow-matching) policies.

1. The Markov Decision Process in Unified Multimodal Reasoning

From the reinforcement-learning perspective, the paper seamlessly converts interleaved multimodal generation into a sequential MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R})$.

Each step $k$ corresponds either to one token prediction in the text stage or to one denoising step in the image stage.

- **State space $\mathcal{S}$**: States evolve through two stages. In the text stage, state $s_k^{\mathrm{txt}}=(c,y_{\lt k})$ contains the input prompt $c$ and all generated reasoning tokens $y_{\lt k}$. In the image stage, state $s_k^{\mathrm{img}}=(c,y,x_{t_k},t_k)$ expands to include the prompt, the complete reasoning-text sequence $y$, the current noisy image latent $x_{t_k}$, and the current flow time $t_k$.
- **Action space $\mathcal{A}$**: In the text stage, action $a_k^{\mathrm{txt}}\in\mathcal{V}$ is a single token sampled from the large language model's vocabulary. In the image stage, action $a_k^{\mathrm{img}}=x_{t_k-\Delta t}\in\mathbb{R}^d$ is the denoised latent predicted for the next flow-matching step.
- **Transition $\mathcal{P}$**: Given an action, transitions in both modalities are deterministic. The text component appends $a_k^{\mathrm{txt}}$ to the sequence, while the image component advances the latent from $x_{t_k}$ to $x_{t_k-\Delta t}$.
- **Reward $\mathcal{R}$**: This is a typical sparse-reward setting. The model receives the final reward $R(x_0,c)$ only when the image latent has been fully denoised to terminal state $x_0$. Rewards for all intermediate reasoning and denoising steps are set to zero.

2. Algorithmic Workflow

**Policy initialization**: Load a unified multimodal foundation model capable of interleaved generation (the study uses the supervised-fine-tuned, or SFT, Bagel model) and denote it as the initial policy $\pi_{\theta_0}$.

**Text-reasoning sampling**: For a given user prompt $c$, the model invokes the autoregressive policy $\pi_{\theta}(a_k^{\mathrm{txt}}\mid s_k^{\mathrm{txt}})$ to generate $G$ distinct reasoning chains in parallel, denoted by $\{y_i\}_{i=1}^{G}$.

**Image flow-matching sampling**: These $G$ reasoning chains then serve as conditioning contexts, guiding the model to generate $G$ corresponding image-denoising trajectories $\{x_i\}_{i=1}^{G}$ through a hybrid SDE–ODE integrator policy $\pi_{\theta}(a_k^{\mathrm{img}}\mid s_k^{\mathrm{img}})$.

**Compute terminal visual rewards**: Evaluate the final $G$ images with a reward model fine-tuned from InternVL, measuring their visual quality and consistency with the user prompt to obtain sparse terminal rewards $R_i$.

**Compute group-relative advantages**: Rather than relying on a traditional value model, compute each sample's relative advantage $\hat A_i$ directly within the generated group. The formula is:

$$
\hat A_i=
\frac{R_i-\mathrm{mean}(\{R_j\}_{j=1}^{G})}
{\mathrm{std}(\{R_j\}_{j=1}^{G})}
$$

Compute text- and image-policy objectives:

- **TextGRPO**: Use the standard GRPO objective to compute the text loss $\mathcal{J}_{\mathrm{text}}(\theta)$, with built-in importance-sampling clipping and a KL-divergence penalty.
- **FlowGRPO**: To introduce the exploration required by reinforcement learning into a deterministic ordinary differential equation (ODE), convert sampling into a stochastic differential equation (SDE). At the same time, use ratio normalization (RatioNorm) to normalize the scale of ratios and ensure that clipping remains effective. This yields the image objective $\mathcal{J}_{\mathrm{flow}}(\theta)$.

$$
\mathcal{J}
=\mathcal{J}_{\mathrm{text}}+\lambda\mathcal{J}_{\mathrm{flow}}
$$

Jointly update the model weights $\theta$ through gradient ascent (the experiments set $\lambda$ to 1 to balance text and image optimization).

For the principles of FlowGRPO, see the related discussion of reinforcement learning for diffusion and flow-matching models.

3. Removing CFG

- Classifier-free guidance (CFG) is currently the most widely used inference technique in diffusion models and flow matching for forcibly improving image–prompt alignment. At each denoising step, the model performs two forward passes: one with the prompt (conditional evaluation) and one without it (unconditional evaluation). The algorithm then amplifies the vector difference between the two outputs, forcibly “pushing” generation toward the prompt.
- **Why abandon it entirely during training?**
  - **A computational disaster**: Although CFG works well during inference, it is disastrous for reinforcement-learning training. In multiconditional generation (such as image editing), CFG may require more than three evaluations per step. Across multiple rounds of interleaved generation, the model must continuously manage and branch multiple contexts.
  - **A gradient maze**: In the RL framework, this multiplied computation not only causes GPU-memory and compute requirements to explode, but, more critically, constructs a complex computational graph with many branches, making gradient computation through backpropagation extremely difficult and inefficient.
- **Linear, unbranched trajectory unrolling**
  - UniGRPO directly removes CFG evaluation branches during training and requires the model to perform only a single, unidirectional conditional computation (that is, linear, unbranched unrolling).
  - **What justifies this?** The authors argue that because the RL reward function already evaluates image–text alignment, maximizing this reward allows the model to directly “internalize” alignment capability in its neural-network weights, fundamentally eliminating dependence on CFG's dual branches.

4. Adjusting the KL Constraint

- **The conventional latent-space KL-divergence trap**
  - To prevent a model from “forgetting” pretrained prior knowledge while pursuing high rewards, the conventional approach penalizes the KL divergence between the current policy and a reference model. In the SDE formula above, this local KL divergence can be computed analytically and exactly.
  - **A critical time-step-dependent loophole**: Mathematical derivation shows that this exact KL divergence is essentially the squared difference between predicted velocities, but it must be multiplied by the inverse noise variance, $1/\sigma_{t_k}^2$.
  - **The optimizer's “aggressive exploitation”**: This is where the problem arises. At the very beginning of image generation (when denoising has just started), the image is almost entirely noise, and the variance $\sigma_{t_k}^2$ is extremely large. Consequently, $1/\sigma_{t_k}^2$ approaches zero. The clever RL optimizer immediately discovers this loophole: it can substantially alter the generation trajectory early on while incurring almost no KL penalty because the weight is so small. This “lawless zone” directly causes severe artifacts and texture breakdown through reward hacking.
- **UniGRPO's solution: unweighted velocity-field MSE**
  - Faced with a mathematically perfect formula that contains an engineering loophole, the authors abandon KL divergence and instead directly compute the mean squared error (MSE) between the model's predicted “velocity field” (the vector-field output $v_{\theta}$) and that of the reference model.

$$
\mathcal{L}_{\mathrm{MSE}}(\theta)
=\lVert v_{\theta}(x_{t_k},t_k,y)-v_{\mathrm{ref}}(x_{t_k},t_k,y)\rVert^2
$$

### (2) VLPO: A GRPO Variant for Continuous Latent Outputs

The standard GRPO objective can be written as:

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\min\left(
r_{i,t}(\theta)\hat A_{i,t},
\mathrm{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat A_{i,t}
\right)
-\beta\mathrm{KL}
\right]
$$

- **Advantage calculation ($\hat A_{i,t}$)**: Exactly the same. For the same question, the model generates a group of responses. The mean and standard deviation of their final scores yield a normalized relative advantage:

$$
\hat A_{i,t}=\frac{r_i-\mathrm{mean}}{\mathrm{std}}
$$

- **PPO clipping**: Exactly the same. Use $1-\epsilon$ and $1+\epsilon$ to limit the magnitude of a single policy update and prevent model collapse.
- **KL-divergence penalty**: Exactly the same. Prevent the new policy from moving too far from the reference model.

The only difference is the definition of the probability ratio $r_{i,t}(\theta)$:

- **For text tokens (GRPO/discrete space)**: Text has an explicit vocabulary distribution, so the ratio is simply the new probability divided by the old probability:

$$
r_{i,t}(\theta)=
\frac{\pi_{\theta}(o_{i,t}\mid\cdots)}
{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid\cdots)}
$$

- **For latents (VLPO/continuous space)**: Latents are high-dimensional continuous vectors without softmax probabilities. Monet assumes that latents follow a Gaussian distribution and derives an equivalent continuous probability ratio:

$$
r_{i,t}(\theta)=
\exp\left(
-\frac{1}{2\sigma^2}
\left\lVert h_{i,t}^{\mathrm{old}}-h_{i,t}^{\theta}\right\rVert^2
\right)
$$

Here, $h_{i,t}^{\mathrm{old}}$ is a frozen feature collected during rollout, and $h_{i,t}^{\theta}$ is a feature produced by the network currently being optimized.

Workflow summary: During reinforcement learning, if a chain of thought containing latents produces a correct result ($\hat A_{i,t}>0$), VLPO uses the same GRPO formula framework and maximizes $r_{i,t}(\theta)$ to pull $h_{i,t}^{\theta}$ in Euclidean space toward the successful $h_{i,t}^{\mathrm{old}}$ (minimizing their L2 distance). This enables an MLLM, for the first time, to use the final reward signal to directly fine-tune its internal continuous visual-feature space.

### (3) GRPO with Unified Multimodal Discretization

LongCat-Next processes all modalities uniformly as discrete token sequences, so GRPO can be used for all of them. However, GRPO is separate for different modalities, and also separate for each of the eight visual levels (with a weighted average taken at the end). They do not share a pi function because their meanings differ.

The KL constraint is removed here, but GRPO's clipping constraint is retained.

Unified multimodal training is difficult and prone to instability. RL for modern large models generally samples several steps first and then trains jointly in batches. If a low-probability noise token is occasionally sampled, positive feedback can cause that trajectory to cascade into garbled output later. Once it is added to the training data, collapse becomes very likely. Two filters are introduced to prevent training collapse:

1. Entropy-based filter: In RL, once a few tokens sample extremely low-probability noise tokens, subsequent generation can cascade into garbled output, giving the whole sequence extremely high entropy. Compute the mean $\mu_H$ and standard deviation $\sigma_H$ of sequence-wise entropy across all sequences in the same minibatch. If $H_{\mathrm{seq}}>\mu_H+n\sigma_H$ for a sequence, it is considered an outlier and discarded without contributing to gradients.
2. Training–inference difference filter: Detect per-token probability differences between the rollout sampling policy and the current training policy. If any token in a sequence exceeds threshold $\delta$, discard the entire sequence to prevent training contamination. The underlying cause is a hardware or numerical-precision mismatch that introduces extremely low-probability tokens during rollout. These tokens have high probabilities during sampling but extremely low probabilities under the current training policy. Standard importance-sampling corrections (such as TIS and MIS) can only average at the token level and cannot handle such extreme single-token anomalies.

## VI. Emergence of World-Model Capabilities

The experiments adopt the navigation world model (NWM) setting and innovatively encode robot navigation actions (such as translation and rotation) directly as discrete natural-language text tokens (such as $dx=+1.338$ or `go on the road`).

The results show that without any special architectural changes, the model can accurately predict future visual states from context frames and text instructions alone. More importantly, this understanding of physical laws mainly comes from earlier large-scale unlabeled video pretraining: training on actual online navigation data needs to reach only 1% of the total data for performance to saturate.

In other words, the fundamental understanding of language and world states elicited by multimodal joint pretraining incorporating world models is the key. On top of this, only a small amount of domain-specific data is needed to reach saturated performance.

## References

- Meituan LongCat Team. (2026). [LongCat-Next: Lexicalizing Modalities as Discrete Tokens](https://arxiv.org/abs/2603.27538). arXiv:2603.27538.
- Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., & Luo, P. (2024). [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848). arXiv:2410.13848.
- Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., Shi, G., & Fan, H. (2025). [Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683). arXiv:2505.14683.
- Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., & Levy, O. (2024). [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039). arXiv:2408.11039.
- Shi, Q., Bai, J., Zhao, Z., Chai, W., Yu, K., Wu, J., Song, S., Tong, Y., Li, X., Li, X., & Yan, S. (2025). [Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model](https://arxiv.org/abs/2505.23606). arXiv:2505.23606.
- Cui, Y., Chen, H., Deng, H., et al. (2025). [Emu3.5: Native Multimodal Models are World Learners](https://arxiv.org/abs/2510.26583). arXiv:2510.26583.
- Diao, H., Wu, P., Deng, H., et al. (2026). [SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture](https://arxiv.org/abs/2605.12500). arXiv:2605.12500.
- Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., & Park, T. (2024). [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828). CVPR.
- Liu, J., Ye, Z., Yuan, L., et al. (2026). [UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation](https://arxiv.org/abs/2603.23500). arXiv:2603.23500.
- Wang, Q., Shi, Y., Wang, Y., et al. (2025). [Monet: Reasoning in Latent Visual Space Beyond Images and Language](https://arxiv.org/abs/2511.21395). arXiv:2511.21395.
