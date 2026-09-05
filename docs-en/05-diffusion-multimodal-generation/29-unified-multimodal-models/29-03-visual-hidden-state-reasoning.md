---
title: "29.3 Reasoning with Visual Hidden States in Chain of Thought"
chapter_title: "Unified Multimodal Understanding and Generation Models"
section_id: "29-03"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/29.统一多模态理解-生成模型/29.3 将视觉隐状态融入CoT的推理.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 29.3 Reasoning with Visual Hidden States in Chain of Thought

## I. CoT Alternating between Text Tokens and Visual Hidden States

The core idea of incorporating visual hidden states into CoT is to stop relying solely on external tools to generate images and instead explicitly maintain a segment of continuous latent representations within the model, allowing reasoning to carry visual space, object structure, and intermediate observations.

### (1) Why Visual Hidden States Are Needed

Traditional methods generally rely on the model calling external tools, such as bounding-box prediction, code interpreters, or depth-estimation models, to generate intermediate visual images. These approaches are too rigid to support flexible, abstract visual thinking like that of humans.

MoNet's solution is to use visual hidden states as intermediate visual thoughts. Instead of generating continuous intermediate images through external tools, the model learns to generate continuous latent embeddings, using them to express its dependence on external image-generation tools.

Visual hidden states primarily support three capabilities:

- **Spatial imagination**: When faced with a task of folding a 2D net into a 3D shape, the model can imagine the complete 3D object in latent space and accurately match the answer.
- **Hallucination mitigation and visual grounding**: The model can internally enlarge and focus on key image regions, such as zooming in on a white pillow or checking whether a cup contains water, to obtain fine-grained information and avoid the detail hallucinations common in traditional models.
- **Multiround visual refinement**: For subtle visual features, such as comparing the shades of color in two regions, the model can alternate visual-thought generation over multiple rounds, gradually focusing and integrating observations to reach an accurate judgment.

Traditional latent approaches face two challenges:

1. Directly aligning thousands of image tokens incurs extremely high computational and memory costs, while simple mean pooling destroys visual-feature details.
2. The next-token-prediction objective during supervised fine-tuning is prone to overfitting, and ordinary reinforcement-learning methods optimize only discrete text tokens, failing to adequately optimize visual hidden states.

## II. Monet: A Model That Replaces Visual CoT with Latents and Its Training

### (1) MoNet Inference Workflow

MoNet is built on Qwen2.5-VL-7B, and its workflow is divided into inference and training.

During inference:

- Given an image and a question, MoNet automatically decides when to generate a special starting token `<latent>` to activate latent visual reasoning.
- Once latent-space mode begins, the hidden representation from the decoder's final layer is fed back into the network as the input embedding for the next step.
- After generating a fixed number $K$ of latent embeddings, the model outputs the ending token `</latent>` and automatically switches back to standard textual reasoning.

The loss for aligning observations and latents can be written as:

$$
L_{\text{align-obs}}
=\frac{1}{N}\sum_i\sum_l
\left(
1-\cos\left(h_{\text{obs}}^{d(i,l)}.\mathrm{detach}(),h_{\text{obs}}^{s(i,l)}\right)
\right)
$$

Here, `detach` blocks gradients on the teacher-observation-feature side, allowing the student's latents to align with the observation features.

### (2) Sequence Structure and Visibility Rules

In the student model's context, the sequence is arranged as follows:

```text
[Question] -> [Original image] -> [Partial reasoning text] -> [Auxiliary image (Aux Img)] -> [<latent> Latents 1...K </latent>] -> [Key observation text (Obs Text)]
```

To force information to undergo funnel-like compression, MoNet modifies the standard causal attention mask and defines strict visibility rules:

1. **Latents can see the auxiliary image (Aux Img)**. The generated latents can directly access current auxiliary-image features through cross-attention or self-attention.
2. **Key observation text (Obs Text) cannot see the auxiliary image (Aux Img)**. This path is masked to prevent the text from reading the auxiliary image directly.
3. **Key observation text (Obs Text) can see the latents**. The text can obtain visual information only indirectly through the latents.

The ingenuity of this mask design is that it cuts off the shortcut through which subsequent text could directly read the auxiliary image. To predict observations accurately, subsequent text must extract information from the preceding latents. This forces the latents to explicitly encode key observations, becoming the sole bridge between explicit visual information and textual reasoning.

The number of latents is fixed, for example, $K=8$. The model uses these eight latents to condense and extract features from the hundreds or thousands of patch tokens in the auxiliary image.

### (3) SFT Stages 2 and 3

MoNet distinguishes two stages in SFT.

- **Stage 2: Knowledge distillation with auxiliary images**. Real auxiliary images are included in the sequence. In this open-book examination, the model directly observes the auxiliary images to generate features, aiming to obtain target latents $h_{\text{latent}}^{(i)}$ containing detailed visual information from the teacher.
- **Stage 3: Latent generation without auxiliary images**. Real auxiliary images are completely removed from the sequence, which becomes something like:

```text
[Question image] -> [Partial reasoning text] -> [<latent> Latents </latent>] -> [Key observation text]
```

The model now enters a closed-book examination and must autoregressively generate latents from its understanding of the preceding context. The training objective is to minimize the cosine distance between the generated latents $\hat h_{\text{latent}}^{(i)}$ and the top student's notes saved in Stage 2, namely the target latents $h_{\text{latent}}^{(i)}$.

To support distillation of complex visual spaces, the authors construct a high-quality dataset containing 125,000 interleaved image–text chains of thought. Its selection process has three stages:

1. **Filter irrelevant samples**: Retain only questions that the original Qwen2.5-VL-7B answers incorrectly without auxiliary images, ensuring that auxiliary images genuinely provide a crucial benefit.
2. **Verify auxiliary images**: Use a larger model, such as Qwen2.5-VL-72B, to verify that the question can be answered correctly after auxiliary images are introduced, excluding ineffective images.
3. **Expand precise supervision signals**: Use frontier large language models such as DeepSeek-V3.1 and Gemini 2.5 Pro to generate complete judgments, carefully extract key tokens in the text that directly depend on intermediate visual observations, and enclose them in `<observation>` tags as latent-space alignment supervision during SFT Stage 2.

### (4) Continuous Latent-Space Updates in Reinforcement Learning

The standard GRPO objective is extended into a policy-update objective:

$$
J_{\text{GRPO}}(\theta) =
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

Here:

- **Advantage calculation**: For the same question, the model generates a group of responses, and their final scores are normalized by the within-group mean and standard deviation to obtain relative advantages $\hat A_{i,t}$.
- **PPO clipping**: Use $1-\epsilon$ and $1+\epsilon$ to limit the magnitude of a single policy update and prevent model collapse.
- **KL-divergence penalty**: Prevent the new policy from moving too far from the reference model.

The only difference is in the definition of the probability ratio $r_{i,t}(\theta)$.

For text tokens, because text has an explicit vocabulary distribution, the new-to-old policy probability ratio can be used:

$$
r_{i,t}(\theta) =
\frac{\pi_{\theta}(o_{i,t}\mid\cdots)}
{\pi_{\theta_{\text{old}}}(o_{i,t}\mid\cdots)}
$$

For continuous latents, there are no softmax probabilities. MoNet adopts VLPO's continuous-space approach, assumes a Gaussian distribution for latents, and derives an equivalent continuous probability ratio:

$$
r_{i,t}(\theta) =
\exp\left(
-\frac{1}{2\sigma^2}
\left\lVert h_{i,t}^{\text{old}}-h_{i,t}^{\theta}\right\rVert^2
\right)
$$

Here, $h_{i,t}^{\text{old}}$ is a frozen feature collected during rollout, and $h_{i,t}^{\theta}$ is a feature generated by the network currently being optimized.

During reinforcement learning, if a chain of thought containing latents obtains a positive advantage, VLPO uses the same GRPO formula framework and maximizes $r_{i,t}(\theta)$ to move $h_{i,t}^{\theta}$ toward the positive sample in Euclidean space, reducing their $L_2$ distance. This enables an MLLM, for the first time, to use the final reward signal to directly fine-tune its internal continuous visual-feature space.

## References

- Li, B., Sun, X., Liu, J., Wang, Z., Wu, J., Yu, X., Chen, H., Barsoum, E., Chen, M., & Liu, Z. (2025). [Latent Visual Reasoning](https://arxiv.org/abs/2509.24251). arXiv:2509.24251.
- Tong, J., Gu, J., Lou, Y., Fan, L., Zou, Y., Wu, Y., Ye, J., & Li, R. (2025). [Sketch-in-Latents: Eliciting Unified Reasoning in MLLMs](https://arxiv.org/abs/2512.16584). arXiv:2512.16584.
- Wang, Q., Shi, Y., Wang, Y., Zhang, Y., Wan, P., Gai, K., Ying, X., & Wang, Y. (2025). [Monet: Reasoning in Latent Visual Space Beyond Images and Language](https://arxiv.org/abs/2511.21395). arXiv:2511.21395.
- Cheng, T., Chen, S.-Z., Zhang, H., Qin, Y., Luo, J., & Wei, Z. (2026). [HyLaR: Hybrid Latent Reasoning with Decoupled Policy Optimization](https://arxiv.org/abs/2604.20328). arXiv:2604.20328.
