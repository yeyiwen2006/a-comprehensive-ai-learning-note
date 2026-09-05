---
title: "20.6 Multi-Expert On-Policy Distillation"
chapter_title: "Optimizing Large-Model Architectures and Training Methods"
section_id: "20-06"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/20.大模型的架构和训练方法优化/20.6 多专家On-policy Distillation.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 20.6 Multi-Expert On-Policy Distillation

DeepSeek-V4 uses multi-expert on-policy distillation: starting from a pretrained model, it first trains multiple domain experts using SFT and RL, then distills their knowledge into a unified student model.

## I. Training Domain-Expert Models

Starting from the pretrained model, train a set of domain-expert models using SFT and GRPO, covering mathematics, coding, agents, instruction following, and other domains. Industry practice also generally trains subversions with different reasoning intensities for each domain, corresponding to Non-think, Think, and Think Max. These three modes use different length penalties and context windows during RL.

Agent experts also incorporate the Quick Instruction mechanism. Chat products involve many auxiliary tasks, such as deciding whether to trigger a search or identifying intent. Traditionally, another small model handles these tasks, requiring prefill to be repeated whenever the context changes. V4 directly appends a set of special tokens to the input sequence, with each token corresponding to an auxiliary task, reusing the existing KV cache and reducing time to first token.

## II. On-Policy Distillation

Use on-policy distillation to combine the knowledge of all experts into a unified student model. Compared with off-policy methods, on-policy distillation better avoids catastrophic forgetting. Specifically, the student learns the output probability distributions of multiple expert models on trajectories it generates itself.

## References

- DeepSeek-AI. (2026). [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348). arXiv:2606.19348.
