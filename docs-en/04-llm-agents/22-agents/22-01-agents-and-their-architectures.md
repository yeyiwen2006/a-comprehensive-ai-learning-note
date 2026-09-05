---
title: "22.1 Agents and Their Architectures"
chapter_title: "Agents"
section_id: "22-01"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/22.智能体/22.1 智能体及其架构.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "all Word images manually transcribed as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 22.1 Agents and Their Architectures

## I. Overview of Agents

An LLM agent is an agent built around an LLM that can interact with users and the external environment (an LLM itself cannot autonomously interact with the external environment), autonomously perceive, perform tasks, and obtain feedback. It extends the LLM beyond traditional "conversation," enabling it to help humans complete real work in actual environments. An agent workflow takes the form of a complete program containing different statement blocks for inputting prompts, calling the LLM, invoking tools, and so on. Achieving the overall agent functionality requires combining model reasoning, tool use, memory retrieval, and other capabilities. During a task, the agent outputs textual thoughts or actions. After an action interacts with the environment, the environmental feedback is fed back into the agent for further reasoning and processing.

A complete agent should contain the following modules:

Reasoning module: a large language model capable of generating action instructions.

Memory or knowledge-base module: apart from the context window (explained below), memory or knowledge bases are generally external to the model at present. The model can use retrieval-augmented generation (RAG) to access their content during generation.

Tool-use module: the core interface for model–environment interaction. Through tool calls, the model acts in the environment and obtains feedback.

An agent's context is its input on each occasion. Unlike an LLM context containing only the prompt and previously generated content, an agent's context also includes environmental feedback (possibly compressed), such as tool-call and retrieval results, and may contain information retrieved from memory. Thus, the paradigm in the agent era is shifting from prompt engineering to context engineering. In agentic RL, the agent's "state" in a Markovian environment also includes all these context components. For input $t+1$ within the same task (dialogue round $t+1$):

Context injection / state transition: concatenate action $a_t$ and feedback $o_{t+1}$ (denoted $\oplus$) to the existing context, completing the so-called "instantaneous transition" to a new decision state:

$$
h_{t+1}=h_t\oplus a_t\oplus o_{t+1}
$$

## II. The ReAct Paradigm

External programs and prompts guide the model to alternate between thinking, acting, and observing (analogous to human thought, action, and perception). Actions include searching externally for information or outputting a final answer and may involve external tools (the large model generates the precise instructions or arguments needed to call them). Observation means processing the information obtained after the current action or observing environmental changes. During reasoning, outputting particular tokens triggers tool-based action or terminates the loop (some later architectures replace direct termination with a dedicated mechanism that checks task completion).

Consider an example:

Question: What is the elevation range of the area to which the eastern region of the Colorado orogeny extends?

Thought 1: I need to search for the Colorado orogeny, find the area to which its eastern region extends, and then find that area's elevation range.

Action 1: Search [Colorado orogeny]

Observation 1: The Colorado orogeny was an episode of mountain building in Colorado and surrounding areas.

Thought 2: It does not mention the eastern region, so I need to look up the eastern region.

Action 2: Lookup [eastern region]

Observation 2: (Result 1 / 1) The eastern region extends to the High Plains and is called the Central Plains orogeny.

Thought 3: The eastern region of the Colorado orogeny extends to the High Plains. I therefore need to search for the High Plains to find their elevation range.

Action 3: Search [High Plains]

Observation 3: High Plains refers to one of two distinct land regions.

Thought 4: I need to search for High Plains (United States).

Action 4: Search [High Plains (United States)]

Observation 4: The High Plains are a subregion of the Great Plains. From east to west, their elevation ranges from 1800 to 7000 feet (550 to 2130 meters).

Thought 5: The High Plains range from 1800 to 7000 feet in elevation, so the answer is 1800 to 7000 feet.

Action 5: Finish [1800 to 7000 feet]

Of course, this is original ReAct, which requires outputs to be prefixed with "Thought/Action/Observation." Later ReAct variants do not all impose that requirement, but the core idea of a loop comprising these three stages remains. Consider OpenAI Codex:

Suppose the terminal input is: add an architecture diagram to the project's README.md.

Step 1: Construct the prompt

Codex first constructs a prompt containing the system prompt (telling the model who it is and what it can do), tools (the tools available to call), context (environmental context such as the current directory and shell), dialogue or output history, and the user's instruction.

Step 2: Model inference

Codex sends the prompt to the model, which decides to call the shell tool and execute catREADME.md.

Step 3: Tool call

Codex receives the model's request, executes the command locally, and reads the contents of README.md.

Step 4: Feedback

The terminal outputs the contents of README.md. In the loop architecture, this does not mean the process ends. Codex appends the command output to the prompt and sends it back to the model.

The agent loop

The model sees the README and reasons again: it might generate a Mermaid diagram or write an ASCII diagram directly, then call a tool to write it to the file. This loop continues until the model considers the task complete and outputs a closing message. The agent checks errors and validates results on its own, becoming an agent capable of sustained work.

## III. Plan First, Then Execute

For complex tasks, planning before execution often improves accuracy. Cursor, Claude Code, and similar systems have a plan mode that prompts the model to formulate a plan first and then execute it through ReAct. The plan can also be maintained dynamically, updated after every action based on observed feedback and execution progress.

This architecture relies on sound planning, which may be difficult for complex tasks. Therefore, in agents such as Claude Code, planning is no longer a one-time task but a loop: in "read-only mode," the model forms a "reason and plan—read or ask the user" cycle, continually revising the plan through file inspection and user interaction.

## IV. Introspection and Reflection

Adding a reflection step to ReAct lets an agent generate self-criticism when execution fails or results are unsatisfactory. This self-reflection is stored as short-term memory to remind it not to repeat the mistake on its next attempt. This greatly improves success rates for complex tasks without retraining the model.

## V. Autonomous Loops

We want agents that can loop autonomously, independently setting subgoals, planning, and executing continuously according to an overall objective. Their workflow is a sustained autonomous loop that dynamically measures the gap to the objective, reflects, generates new subtasks, and adjusts execution order until the objective is completed. Challenges include gradual drift from the original objective requiring correction mechanisms; the need for reasonable termination criteria; high ongoing costs; and weak interpretability, potentially even creating safety risks.

## VI. Ralph Loop

In July 2025, software engineer Geoffrey Huntley described Ralph as a minimalist looping coding-agent workflow, whose pure form is a single Bash command: `while :; do cat PROMPT.md | claude-code ; done`. Each iteration rereads the same task prompt and starts a new agent process. External state such as code changes, test results, and Git history can persist, but the full conversation context of the preceding iteration does not automatically carry over. The original Ralph loop itself contains neither a "completion flag" nor a stop hook. Some later tools add programmable completion conditions and stop hooks to this idea: if tests fail or completion criteria are not detected, the hook prevents termination and triggers another iteration. These later wrappers should be distinguished from Huntley's one-line loop.

Ralph Loop can address many pain points that cause ordinary AI tools to fail. Most people using AI coding tools have an idea, but the task is too large for the AI to solve in one context window. Ralph does not require solving everything at once. We can therefore have the AI plan first, then break everything within a loop into the smallest units that "the AI can complete in one attempt and judge as correct or incorrect in one attempt." Specifically:

1. Enter clear instructions.

2. Have the AI decompose the task into a requirements list. Each task needs an explicit check for success. Examples are "Add a priority column, defaulting to medium" and "Show the options All, High, Medium, Low in the dropdown," rather than vague wording such as "make it look good." The description of the requirements directly determines output quality.

3. Run the Ralph loop.

## References

- 温睦宁、林江浩、张伟楠、俞勇. (2025). [*Hands-on Learning of Large-Model Agents*](https://haa.boyuai.com/) (translated title; in Chinese). Posts & Telecom Press. ISBN 978-7-115-68638-1.
- Yao, S., Zhao, J., Yu, D., et al. (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629). ICLR 2023.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). NeurIPS 2023.
- Huntley, G. (2025). [Ralph Wiggum as a “software engineer”](https://ghuntley.com/ralph/).
- Bolin, M. (2026). [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/). OpenAI.
