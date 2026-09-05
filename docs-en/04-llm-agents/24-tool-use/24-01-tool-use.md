---
title: "24.1 Tool Use"
chapter_title: "Tool Use"
section_id: "24-01"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/24.工具调用/24.1 工具调用.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "not applicable; source contains no Word images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 24.1 Tool Use

To turn a large language model from merely a "brain" into an agent that perceives and acts in real time, we need to give it the ability to call tools. This includes the following aspects.

## I. Basic Workflow and Strategies

1. Basic workflow: determine whether a tool is needed—select a suitable tool—call the tool—parse its output.

2. Strategies

(1) Prompt-driven: for example, add "If you think your current knowledge is insufficient or real-time information is needed, use the appropriate tool. If you can answer directly, generate the answer." Available tool definitions are usually included in the prompt. Given limited context, in practice, usually only "meta-tool" definitions (such as Bash commands that let the model invoke secondary tools) or a tool list are included, letting the model search the filesystem for tools as needed.

(2) The ReAct paradigm (reason + act)

Earlier, we discussed prompting models to use tools and chains of thought for reasoning. These operate in parallel without good coordination. ReAct combines them, as in this example:

Question: How many Japanese yen equal one US dollar?

Thought: I do not know the current exchange rate, so I should look it up first.

Action: SEARCH[USD to JPY exchange rate]

Observation: The current exchange rate is 157.5.

Thought: I have the exchange rate and can now calculate.

Action: CALC[157.5 × 20]

Observation: The result is 3150.

Final Answer: 20 US dollars is approximately 3150 Japanese yen.

Of course, this is still essentially implemented through prompts.

(3) Train with RL so that the model autonomously learns when to call tools.

## II. Tool Categories

1. Information-retrieval tools: analogous to human "sensory neurons," they help the model obtain information or knowledge in real time, such as search engines, knowledge databases, and uploaded-document parsers.

2. Computation and reasoning tools: analogous to "central neurons," they support precise calculation or reasoning, avoiding errors that probability-based models may make when reasoning on their own. Examples include calculators and code interpreters.

3. Device-control tools: analogous to "motor neurons," they help models perform tasks in real environments, such as tools controlling operating systems, applications, or robots.

## III. Tool Interfaces

For the model to understand a tool's purpose and invocation method, we need specific API interfaces that encapsulate tools in a comprehensible form. An interface needs a tool name, purpose description, input parameters, and an explanation of which parameters are required.

## IV. Integrating Multiple Tools

1. Tool selection: provide a tool list in the prompt or configure keyword-triggered routing.

2. Priorities and fallbacks: prefer tools with higher success rates or lower costs; if a call fails, switch to a lower-priority tool. After several failures, answer directly to avoid an infinite loop. Reflection can also encourage the model to assess failure causes and improve its next call.

## V. Compressing Tool Results

Tool results can be enormous and fill the context. Pruning can compress earlier outputs before passing them to the LLM. This is lossy compression, but it does not rewrite the result files on disk.

## VI. Reasoning-Context Management: Preventing Broken Reasoning after Tool Calls (DeepSeek, 2025)

A common workflow for reasoning models with tool use is: user question → model's "internal reasoning" (state A) → tool execution → tool result + user question → model's "internal reasoning" (state B). State A is not retained when entering B.

DeepSeek designs a mechanism that lets a model complete complex tasks by "consulting notes" like a human. The V3.2 workflow is: user question → model generates reasoning\_content\_A (on "scratch paper") → tool execution → tool result + user question + reasoning\_content\_A (as input context) → model continues reasoning from the previous draft and generates reasoning\_content\_B.

## References

- 温睦宁、林江浩、张伟楠、俞勇. (2025). [*Hands-on Learning of Large-Model Agents*](https://haa.boyuai.com/) (translated title; in Chinese). Posts & Telecom Press. ISBN 978-7-115-68638-1.
- Yao, S., Zhao, J., Yu, D., et al. (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). ICLR 2023.
- DeepSeek-AI. (2025). [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556). arXiv:2512.02556.
