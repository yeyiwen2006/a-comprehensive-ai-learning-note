---
title: "33.1 Overview of World–Action Models"
chapter_title: "World–Action Models"
section_id: "33-01"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/33.世界-动作模型/33.1 世界-动作模型概述.docx"
status: "manually rebuilt and checked against Word"
ocr: "all Word-visible text and formula images manually transcribed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 33.1 Overview of World–Action Models

## I. The Concept of a World–Action Model

A world–action model (WAM) applies the world-model paradigm to embodied AI foundation models. It combines state prediction and action generation, taking past states (and actions already taken) as input and using a unified model to predict the next state and subsequent actions, sequentially or simultaneously. A unified model is used because both tasks fundamentally learn physical-world laws, many parameters overlap, and unification better follows the end-to-end principle.

## II. Training Objectives of World–Action Models

A WAM's loss is generally a weighted sum of state loss (video pixels or visual features) and action loss:

$$
\mathcal{L}_{\mathrm{vid}}+\lambda_{\mathrm{act}}\mathcal{L}_{\mathrm{act}}
$$

Both usually use flow-matching losses:

For target variable $y$ (either an action sequence $a_{1:H}$ or future visual features $z_{1:T}$), sample Gaussian noise $\varepsilon\sim\mathcal{N}(0,I)$ and time step $t\in(0,1)$ to construct an interpolated sample:

$$
y_t=(1-t)y+t\varepsilon.
$$

The model predicts the velocity field through a standard flow-matching loss:

$$
\mathcal{L}_{\mathrm{FM}}(y)=\mathbb{E}_{y,\varepsilon,t}\left[\left\|f_\theta(y_t,t,o,l)-(\varepsilon-y)\right\|_2^2\right].
$$

## III. Embodiment-Native Models

Early WAMs were often obtained by post-training video-generation models. However, video-generation models were designed for digital-content creation, and transferring them to robots creates three problems:

1. The representation VAE is optimized only for pixel-level reconstruction. Much of its valuable latent space is devoted to lighting, background textures, and surface highlights, while control-relevant events such as a cup moving left occupy only a tiny fraction. It preserves visual details while discarding physical and semantic structure. The action module is also often attached externally, leaving world states and actions in two misaligned spaces.

2. High-dimensional video tokens combined with multistep iterative denoising create long latency, while real-robot operation requires the policy to read new observations at every action-chunk boundary and correct itself as needed.

3. Internet videos lack action labels, and generic video-prediction objectives do not teach the model “how actions change the world.”

Many WAMs are therefore shifting toward native training on embodied data. This demands more compute and data but also raises the ceiling of model capability.

## References

- Wang, S., Shi, J., Fu, Z., et al. (2026). [World Action Models: The Next Frontier in Embodied AI](https://arxiv.org/abs/2605.12090). arXiv:2605.12090.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR.
