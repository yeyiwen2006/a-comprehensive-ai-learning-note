---
title: "22.2 Prompt Engineering"
chapter_title: "Agents"
section_id: "22-02"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/22.智能体/22.2 提示工程.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "not applicable; source contains no Word images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 22.2 Prompt Engineering

## I. Basic Concepts

A prompt is a language instruction provided to a large language model or agent. It specifies the task we want the model to complete and its requirements. Prompts are generally divided into two categories.

1. System prompt: an initial set of instructions or background information given to the model to guide the agent's behavior and response patterns, defining its role, tone, scope of knowledge, and so on.

2. User prompt: the user's question, request, or command, intended to obtain a specific response from the model or complete a particular task. It is the direct trigger for the LLM agent's response.

In multi-turn question answering, putting unchanged rules in the system prompt and changing rules in the user prompt is much more convenient.

## II. What a Prompt Should Include

1. Task description: explicitly tell the model what task to complete.

2. Background: the current task, agent identity, or contextual environment.

3. Output constraints: output format, style, or word count.

4. Examples (optional): provide reference examples for the model to imitate.

## III. Principles of Prompt Design

1. Clarity: avoid vague or ambiguous instructions.

2. Specificity: provide concrete, detailed constraints whose correctness is preferably easy to judge.

3. Background or role setting: for example, "Assume you are an experienced doctor" leads the model to provide more specialized medical judgments; "Assume you are a senior lawyer" makes it focus more deeply on law.

4. Moderate redundancy: particularly important requirements can be emphasized repeatedly.

## IV. Prompt Engineering Techniques

1. Chain of thought

Prompt the model to generate intermediate reasoning steps resembling human thought before producing its final answer, decomposing complex reasoning into simpler, less error-prone subtasks.

Why does it work?

(1) Reduced cognitive load: a large language model is essentially a probability-based sequence predictor. Directly predicting the final answer (such as "108") is extremely difficult because the "semantic distance" from question to answer is very large. CoT inserts intermediate steps, turning a "long-range dependency" into several "short-range dependencies." The model only needs to predict the next most reasonable reasoning step or symbol based on the preceding context, better matching its underlying capabilities.

(2) Alignment with model strengths: models learn grammatical structures, mathematical formulas, logical relationships, and common knowledge from enormous quantities of code and text. By guiding outputs such as "first," "second," and "according to formula XX," CoT activates strong capabilities in procedural knowledge and logical structure rather than forcing the model to "guess" a number.

(3) Fewer compositional generalization errors: complex problems combine basic concepts such as addition, multiplication, and formulas. A model may know each basic concept but make mistakes when combining them. CoT makes the composition process explicit, letting the model apply the concepts step by step and improving accuracy.

(4) Emergent capability: CoT is a typical emergent capability. Smaller models (such as those with hundreds of millions of parameters) gain only limited improvements even with CoT prompting. In large models (such as those exceeding 100 billion parameters), this capability suddenly becomes pronounced and effective.

2. In-context learning: adding examples to the input improves response quality and makes answers better match requirements. Specific prompting methods include zero-shot, one-shot, and few-shot.

3. Self-consistency: generate multiple times and choose the most frequent answer, which is often more reliable than a single response.

4. Reflection prompting

We use three models:

(1) Actor: generates text or actions from the state and observations, acts in the environment, and observes the results to obtain a trajectory. A memory component provides additional context, such as past errors.

(2) Evaluator: evaluates the actor's outputs and assigns a reward based on its action trajectory.

(3) Self-reflection: provides feedback and suggestions on the agent's past actions based on the trajectory and reward score, storing them in memory for future reference.

## V. The Evolution of Prompt Engineering

As model capabilities improve, some prompt-design techniques gradually lose their value. Prompt engineering is increasingly incorporated into later approaches such as context engineering and harness engineering. Clearly communicating requirements and specifying boundaries, however, remain necessary.

## References

- 温睦宁、林江浩、张伟楠、俞勇. (2025). [*Hands-on Learning of Large-Model Agents*](https://haa.boyuai.com/) (translated title; in Chinese). Posts & Telecom Press. ISBN 978-7-115-68638-1.
- Wei, J., Wang, X., Schuurmans, D., et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). NeurIPS 2022.
- Wang, X., Wei, J., Schuurmans, D., et al. (2023). [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://openreview.net/forum?id=1PL1NIMMrw). ICLR 2023.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). NeurIPS 2023.
