---
title: "20.4 Attention Residuals"
chapter_title: "Optimizing Large-Model Architectures and Training Methods"
section_id: "20-04"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/20.大模型的架构和训练方法优化/20.4 Attention Residuals.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 20.4 Attention Residuals

> This article is a paper-reading note. Its content represents the corresponding paper's method or the author's understanding and should not be treated directly as field consensus or engineering best practice.

## I. Background

Residual connections inside mainstream large models add each layer's processed result to its original input and pass the sum to the next layer. After accumulation at each layer, information is compressed into a mixed state. This mixed state grows increasingly bloated at later layers. As the network deepens, its numerical values grow larger, causing PreNorm dilution: the deeper the layer, the more its contribution is diluted. Experiments show that removing quite a few layers from a very deep model barely changes its performance. We are actually wasting substantial computation.

## II. The Core Idea

In current mainstream architectures, each layer's input = the network input + the outputs of all preceding layers (with equal weights). The Kimi team proposes that each layer instead use the network input + the outputs of all preceding layers (summed with weights according to need), with those weights learned by the layer itself.

Specifically, attention lets the model decide which layers to attend to. Each layer has its own small vector (only one, making it extremely lightweight), which is used to calculate similarities with the outputs of all preceding layers. Higher similarity gives a larger weight, lower similarity a smaller weight, and the results are summed with these weights. Notably, attention layers and MLP layers can attend to different historical layers.

In this way, information from early layers is not overwhelmed and can be retrieved as needed. The total number of added parameters is negligible relative to a large model with billions of parameters, but performance improves to 1.25 times the original, equivalent to saving 20% of the computation at the same performance.

## III. Block Attention Residuals

The engineering problem is that each layer must store the outputs of all preceding layers in order to attend to them. Modern large models generally have over a hundred layers, and during training they are split into many parts distributed across different machines (pipeline parallelism). Every machine must maintain all layer outputs, so memory and communication pressure increase dramatically.

Kimi's solution is to store groups rather than individual layers. All layers are divided into several blocks; within each block, outputs are still accumulated in the traditional way, but the entire block is compressed into a vector and stored. Attention then selects across blocks. Storage therefore changes from being proportional to the number of layers to being proportional to the number of blocks. Experiments find that dividing the model into approximately 8 blocks recovers most of the effectiveness of full attention residuals, while requiring only a small fraction of the original memory and communication overhead.

## References

- Kimi Team. (2026). [Attention Residuals](https://arxiv.org/abs/2603.15031). arXiv:2603.15031.
