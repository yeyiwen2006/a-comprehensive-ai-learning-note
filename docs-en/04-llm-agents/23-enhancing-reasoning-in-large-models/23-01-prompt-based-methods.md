---
title: "23.1 Prompt-Based Methods"
chapter_title: "Enhancing Reasoning in Large Models"
section_id: "23-01"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/23.大模型的推理增强/23.1 基于提示的方法.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "not applicable; source contains no Word images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 23.1 Prompt-Based Methods

Reasoning capabilities are becoming increasingly important in large models. Currently, model reasoning takes the form of continually generating tokens (tokens are also generated when a model displays that it is thinking, but they are wrapped in special markers and are therefore also billed in API calls), explicitly advancing logic through "self-talk." The following are prompt-based methods for enhancing reasoning.

## I. Chain of Thought

Ask the model to think step by step, breaking a "large problem" into "small steps." This shortens the distance each step must "cross" in embedding space, enabling complete reasoning by repeatedly "predicting the next step."

Chain of thought is the foundation of large-model reasoning techniques. Its limitations are that each generated step infers only "forward" from preceding content, without working backward from "later objectives" to "earlier required steps"; it lacks self-correction and reflection mechanisms, allowing errors to accumulate; and it follows one path without exploration, making complex, diverse reasoning paths difficult to cover. These three limitations correspond to the three types of methods discussed later.

## II. Task Planning

1. Plan–execute paradigm

Explicitly separate reasoning into "planning" and "execution."

Planning stage: rather than immediately calculating, the model first generates a structured plan from the original problem, listing the key steps required. This constructs an overall guiding framework.

Execution stage: the model uses each plan step as a separate input, generating chains of thought and results step by step.

Advantage: "overview first, steps second" reduces omissions and calculation errors in intermediate stages, significantly improving adaptation to unseen tasks, particularly in zero-shot settings.

2. Subtask decomposition

For complex problems whose difficulty exceeds the examples, such as multistep mathematical reasoning, a least-to-most progression can be used. Decompose a complex problem into subproblems, arrange them in increasing difficulty, and solve them in sequence. When solving problem k, provide the answers to the first k-1 problems as context.

This establishes explicit dependencies across steps, using preceding results to support current reasoning and prevent logical breaks.

## III. Dynamic Adjustment and Reflection

If task planning lets the model "look ahead," dynamic adjustment and reflection let it "look back," reflecting on and correcting errors in past reasoning trajectories.

Initially, prompts implemented this behavior. Later, models learned directly through reinforcement learning to incorporate dynamic adjustment and reflection into their reasoning trajectories.

## References

- Wei, J., Wang, X., Schuurmans, D., et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). NeurIPS 2022.
- Wang, L., Xu, W., Lan, Y., et al. (2023). [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models](https://aclanthology.org/2023.acl-long.147/). ACL 2023. DOI: 10.18653/v1/2023.acl-long.147.
- Zhou, D., Schärli, N., Hou, L., et al. (2023). [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://openreview.net/forum?id=WZH7099tgfM). ICLR 2023.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). NeurIPS 2023.
