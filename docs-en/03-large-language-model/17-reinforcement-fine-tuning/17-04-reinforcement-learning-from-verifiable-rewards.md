---
title: "17.4 Reinforcement Learning from Verifiable Rewards"
chapter_title: "Reinforcement Fine-Tuning"
section_id: "17-04"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/17.强化微调/17.4 基于可验证奖励的强化学习.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 17.4 Reinforcement Learning from Verifiable Rewards

## I. Core Idea

Reinforcement learning from verifiable rewards (RLVR) is an important reinforcement fine-tuning paradigm that has brought a major advance in large-model reasoning. For a pretrained model, rewards based on correctness or automatically verifiable scores—such as mathematical answers or programming outputs—can elicit logical reasoning, allowing it to independently explore strategies for obtaining correct answers.

In principle, LLM scoring is not introduced here, to avoid reward hacking.

## II. Extracting Answers

Model outputs generally contain a long chain of thought, with reasoning inside `<think>...</think>` tags and the final result inside `<answer>...</answer>` tags. By default, the answer is taken to be the final LaTeX string in the final result. If the model fails to follow the format early in RL training (for example, omitting `<answer>` tags), the system marks it wrong (Reward=0) or gives a very small penalty. To receive rewards, the model must therefore learn to obey the format before learning to solve the problem correctly.

For mathematical problems with multiple answer forms, symbolic formula-verification tools can judge equivalence. For programming, code can be executed and rewarded according to its results. Semi-open questions such as “write a molecular formula satisfying these conditions” may have infinitely many answers and can use a dedicated checker script.

## III. Improving Readability

1. Quantification through hard rules: readability first requires format compliance.

(1) Structural completeness: are `<think>` and `<answer>` tags present and closed? Do they contain content?

(2) Repetition: measure n-gram repetition rates. Pure RL models can easily enter loops when stuck, repeatedly producing text such as “therefore...therefore...therefore....” The system detects n-gram repetition frequency and gives a large negative reward above a threshold.

(3) Language consistency: use confidence from a language detector such as fastText. If the prompt is Chinese but the chain of thought suddenly switches to English or gibberish (mixed Chinese and English often appeared early in DeepSeek-R1-Zero), readability is considered poor. The system calculates the proportion of tokens in the target language; a lower proportion incurs a larger penalty.

(4) Length: count tokens to prevent verbose nonsense produced to pad the response or over-elaborate a derivation. A soft length range is usually imposed, with a small negative reward for exceeding it.

2. Model-based quantification: a reward model (RM)

Higher-level readability—logical flow, attractive formatting, and a human-like tone—cannot readily be captured by rules, so another model scores it. This follows the idea of RLHF.

During early training of pure reasoning models such as o1 or R1, this component has little or no weight. Excessive emphasis on readability can harm reasoning ability, as the model may simplify complex reasoning to please humans. It becomes more important later.

3. The most important method: an SFT “cold start” followed by a KL-divergence constraint

Although counterintuitive, most RLVR models inherit their readability from supervised fine-tuning (SFT), rather than learning it during RL. Before RL begins, they are fine-tuned on a small amount of high-quality “ideal reasoning” written by humans or strong models, with excellent formatting, clear logic, and polished language. During RL, the system computes KL divergence between the current policy model and the SFT reference model.

If the model begins speaking incoherently to solve a problem, reducing readability, KL divergence rises sharply and lowers the total reward. Readability is thus implicitly quantified as similarity to human writing conventions.

4. Real-world example: the lesson of DeepSeek-R1-Zero

DeepSeek's paper provides a striking counterexample showing that RL can destroy readability without dedicated constraints. DeepSeek-R1-Zero had neither an SFT cold start nor a readability reward. Pure RL made it more capable and sharply improved problem-solving accuracy, but its outputs became extremely difficult to read.

## References

- DeepSeek-AI. (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- 温睦宁、林江浩、张伟楠、俞勇 (2025). [*Hands-on Learning of Large-Model Agents* (translated title; in Chinese)](https://haa.boyuai.com/). Posts & Telecom Press. ISBN 978-7-115-68638-1.
