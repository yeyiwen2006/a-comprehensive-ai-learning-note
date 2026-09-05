---
title: "28.8 Reasoning Implicit in Video Generation Diffusion"
chapter_title: "Multimodal Generation"
section_id: "28-08"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/28.多模态生成/28.8 视频生成扩散过程中隐含的推理.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 28.8 Reasoning Implicit in Video Generation Diffusion

## I. How the Diffusion Process Performs Reasoning

Previously, the prevailing assumption was that video-model reasoning followed a chain of frames (CoF), unfolding chronologically as video frames play. This study instead proposes and validates a **chain of steps (CoS)** mechanism. Because bidirectional attention in diffusion can cover the entire video sequence, the model actually reasons globally over all frames simultaneously at every denoising step and continually revises them.

### Derivation of the mechanism

Within the mathematical framework of diffusion models based on flow matching, the continuous evolution of latents between noise and clear data is:

$$
x_s=(1-s)x_0+s x_1
$$

Here, $x_0$ is a clean data latent and $x_1$ normally distributed noise. The model learns a velocity field $v_\theta(x_s,s,c)$ during training, allowing it to estimate the current clear result $\hat{x}_0$ at any denoising step $s$:

$$
\hat{x}_0=x_s-\sigma_s\cdot v_\theta(x_s,s,c)
$$

By decoding $\hat{x}_0$ at different diffusion steps, the authors observe two characteristic CoS exploration workflows.

### 1. Multi-path exploration

In early denoising, the model behaves like classical breadth-first search (BFS). It instantiates and explores several candidate reasoning trajectories simultaneously in latent space, for example having a robot in a video take both branches of a maze at once. As denoising advances, suboptimal paths are progressively “pruned” and suppressed until the process converges to the single correct route in later steps.

### 2. Superposition-based exploration

For spatial alignment or object-ordering tasks, the model does not commit early to one state. Instead, it renders multiple mutually exclusive logical hypotheses directly on top of one another, such as ghost images of the same object at different angles. As noise decreases, the overlapping shadows gradually collapse into a final logically consistent entity.

To further show that CoS is dominant, the authors conduct interference experiments with sharply contrasting results:

| Interference mode | Mechanism | Effect on benchmark performance |
| --- | --- | --- |
| Noise at Step | Inject strong noise into all frames at one particular denoising step. | Severely interrupts the reasoning trajectory, dropping the VBVR-Bench score sharply from 0.685 to below 0.3. |
| Noise at Frame | Inject strong noise at one particular frame throughout all denoising steps. | Performance decreases only slightly. The model repairs itself in later steps using context from other frames, leaving reasoning unobstructed. |

## II. Reasoning Capabilities and Their Architectural Specialization

As model and data scale increase, for example in large models such as Wan2.2, the authors observe emergent reasoning properties resembling those of LLMs:

- **Working memory**: in tasks involving occlusion or objects leaving the frame, early diffusion steps establish persistent “memory anchors,” preserving the physical-state continuity of occluded objects in later frames and addressing object permanence.
- **Self-correction and enhancement**: the model exhibits “aha moments” during diffusion. If it initially generates an incorrect shape or incomplete ballistic trajectory, after several steps it can overturn its earlier hypothesis and correct it into a complete, structurally sound final answer.
- **Perception before action**: the model develops a general protocol for video reasoning. Early denoising prioritizes semantic localization, finding the key objects described in the prompt; later steps add action planning, kinematic changes and complex physical interactions.

To seek architectural evidence, the study extracts hierarchical token activations from a single DiT forward pass and finds strong internal functional specialization:

- **Shallow layers (0–9)**: primarily perception, with attention spread across global structure and environmental background.
- **Middle layers (approximately 9–30)**: activations become highly concentrated on semantically relevant foreground objects, with many reasoning features related to motion and interaction. In a key **causal latent-feature swapping experiment**, replacing layer 20's latent variables with another object's representation completely reverses the final logical reasoning result. This conclusively demonstrates that middle layers are the core reasoning region.
- **Deep layers (late layers)**: consolidate and integrate the implicit representations after reasoning to produce the final required pixel-level states.

## III. Reasoning Ensemble Algorithm

1. **Multiple initializations**: assign the same model 3 different random-noise seeds and launch independent forward-inference processes.
2. **Lock onto the critical step**: intercept the model at the first diffusion step ($s=0$), the most important step for determining the reasoning trajectory.
3. **Extract the active window**: from each of the 3 processes, extract the “reasoning-active layers” of the DiT backbone, layers 20–29, with implicit representations denoted by $U^{(i)}$.
4. **Spatiotemporal ensemble voting**: average the three versions of latent variables across space and time. This is an “expert vote” in latent space that filters seed-specific noise.
5. **Guide generation**: continue generation using the stable averaged latents. Experiments show that this simple ensemble strategy directly raises the absolute VBVR-Bench reasoning score by 2%.

## References

- Wang, R., Cai, Z., Pu, F., et al. (2026). [Demystifying Video Reasoning](https://arxiv.org/abs/2603.16870). arXiv:2603.16870.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR.
- Wiedemer, T., Li, Y., Vicol, P., et al. (2025). [Video Models are Zero-Shot Learners and Reasoners](https://arxiv.org/abs/2509.20328). arXiv:2509.20328.
- Wang, M., Wang, R., Lin, J., et al. (2026). [A Very Big Video Reasoning Suite](https://arxiv.org/abs/2602.20159). arXiv:2602.20159.
