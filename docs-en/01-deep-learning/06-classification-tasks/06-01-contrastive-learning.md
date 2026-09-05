---
title: "6.1 Contrastive Learning"
chapter_title: "Classification Tasks"
section_id: "06-01"
language: en
source_language: zh
source_docx: "第1部分 深度学习/6.分类任务/6.1 对比学习.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 6.1 Contrastive Learning

## I. Contrastive Learning

Learn a feature space in which similar data points ("positive pairs") are close together and dissimilar points ("negative pairs") are far apart.

1. Constructing pairs

Apply different augmentations to the same data point, creating positive pairs that look different but have identical semantics. Examples include random cropping, random color perturbation, Gaussian blur, rotation, and flipping of an image, or summarizing a text passage. Samples from different sources form negative pairs.

2. Loss function

Use the cosine between feature-space vectors as similarity. For a sample, compute similarity with the sample derived from itself (the positive pair) and all similarities with other samples (negative pairs). Softmax produces a score that we want to maximize.

Suppose there is a query $q$, a relevant document $d^+$, and $N$ irrelevant documents $d^-$.

We use Information Noise Contrastive Estimation (InfoNCE) loss. This is currently the most widely used formula for embedding training:

$$
\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(q,d^+)/\tau)}{\sum_{i=0}^{N}\exp(\mathrm{sim}(q,d_i)/\tau)}
$$

where:

- $\mathrm{sim}(u,v)$ is cosine similarity: $\mathrm{sim}(u,v)=\frac{u^Tv}{\lVert u\rVert\lVert v\rVert}$.
- $d^+$ is the positive sample.
- $d_i$ includes $1$ positive and $N$ negative samples.
- $\tau$ is the temperature coefficient, controlling sensitivity to difficult samples.

## References

- Oord, A. van den, Li, Y., & Vinyals, O. (2018). [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748). arXiv:1807.03748.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html). ICML 2020.
