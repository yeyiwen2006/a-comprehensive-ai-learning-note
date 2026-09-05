---
title: "22.5 Coding Agent Frameworks"
chapter_title: "Agents"
section_id: "22-05"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/22.智能体/22.5 编程智能体框架.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "not applicable; source contains no Word images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 22.5 Coding Agent Frameworks

An agent harness, generally an agent framework, provides the model with context, tools, a filesystem, a terminal, permissions, memory, and task state, and continually runs the cycle "model reasoning—tool call—feedback—further reasoning." Coding is one of its main applications. The following introduces representative coding agent frameworks with different product forms and design philosophies.

## I. Cursor

Cursor is an AI-native code editor developed by Anysphere, presented as an "AI-first code editor." It retains an IDE form similar to VS Code while deeply integrating AI into code editing, codebase search, terminal execution, debugging, testing, and Git workflows. Users can choose different large language models, and Cursor configures suitable instructions, context, and tools for each.

(I) Main features

1. Combining code inspection with agent use: in Cursor, users can still inspect and modify code as traditional programmers do, while gradually delegating complete tasks to the agent.

2. Multi-model design: its competitiveness does not depend entirely on one foundation model, but substantially on code retrieval, tool design, the execution environment, and the harness itself. The other side is that a given model may perform worse in Cursor than in its provider's own agent framework, because the latter may have been trained jointly with the model during reinforcement learning.

(II) Core architecture

1. Model layer: Cursor is a multi-model agent harness. Users can select different models, while Cursor adjusts system instructions and tool definitions to make them better suited to software engineering.

2. Code-context system: the agent searches the codebase, reads relevant files and directories, and constructs context by combining project rules, conversation history, and the editor's current state. The core issue is not simply "putting the whole codebase into the model," but identifying which parts of hundreds of thousands or more tokens are truly relevant to the current task.

3. Tool-execution system: Cursor Agent can search files, edit code, call the terminal, run tests, search the web, and access external systems through MCP, plugins, and other integrations. The model therefore returns not only text but also tool-call instructions such as "read this file," "modify this code," or "run npm test."

4. Local agents and cloud agents: ordinary agents work directly in the user's development environment, while cloud agents run in independent cloud virtual machines. Each cloud agent has a code repository, dependencies, and authorized network access and credentials. It can run for tens of minutes or hours, continuing even after the user's computer is shut down.

5. Long-running tasks and multi-agent scheduling: a main agent can maintain a long-term objective and assign subtasks to multiple subagents. Cloud agents can also subscribe to PRs, Slack messages, or scheduled events and restart automatically when something new occurs.

## II. Claude Code and OpenAI Codex

Claude Code is an agentic coding tool developed by Anthropic. Initially a terminal-based coding agent, it lets users describe tasks in natural language to perform various programming jobs. It later expanded to VS Code and desktop interfaces, all running the same Claude Code agent engine underneath. Unlike Cursor, Claude Code departed from the traditional editor interaction model from the outset, making the agent the center of the product.

Similarly, OpenAI's Codex is a complete coding agent product and agent harness. It runs in terminals, IDEs, and the cloud and can be embedded into other software through an SDK or App Server; its desktop version has been merged into the ChatGPT app. Codex helps the model obtain context, call tools, maintain state, execute tasks, control permissions, and work continuously across multiple turns.

(I) Main features

1. Agent-centered: the center of the Claude Code and Codex interfaces is a conversation box rather than a code file. An important founding design principle was that "users do not need to inspect code directly."

2. Terminal integration: they can accept other commands' output through pipes or be invoked by shell scripts, CI/CD systems, and other programs. They are therefore not merely chat interfaces, but intelligent programs composable into traditional development toolchains.

3. Comprehensive functionality: CLAUDE.md/AGENTS.md, skills, hooks, MCP, and subagents correspond to long-term instructions, reusable capabilities, deterministic automation, external tools, and multi-agent collaboration, respectively, making them useful examples for understanding agent frameworks.

(II) Core architecture

1. Agent engine: the core is a continuously running agent loop. The model analyzes the current state and decides whether to read files, modify code, or execute commands, then uses tool results for the next reasoning round.

2. CLAUDE.md/AGENTS.md: Markdown files communicate project architecture, coding standards, test commands, and development conventions. Claude Code/Codex reads these files at the start of each session.

3. Memory: the system can record user preferences, previously corrected issues, and project knowledge difficult to infer directly from source code, then reload them in later sessions. These memories are part of model context, not programmatically enforced rules.

4. Tools and MCP: Claude Code/Codex includes filesystem, shell, Git, and other development tools for reading and editing code, running local development tools, executing commands, and so on. It can also access external systems, such as design documents and databases, through the Model Context Protocol (MCP).

5. Skills and hooks: skills package common workflows as reusable capabilities. Hooks automatically run programs before or after particular agent actions, such as running a formatter after every file modification or enforcing certain operations before a commit.

6. Subagents and agent teams: complex tasks can be split among agents with independent contexts. For example, one analyzes the backend, another the frontend, and a third handles testing, with the main agent integrating the results.

7. Sandbox and approval: manage "what the system permits the model to do," such as modifying current-workspace files, accessing the network or crossing working-directory boundaries, and when additional authorization is required.

8. SDK and App Server: for Codex, the SDK lets programs start, pause, and resume Codex agents. The App Server further exposes the full agent lifecycle, including threads, turns, tool events, streaming outputs, and permission requests.

## III. OpenClaw

As the most representative agent project in the open-source community at the beginning of 2026, OpenClaw is a free, open-source local AI agent runtime that can be deployed directly on a computer. It does not contain a large language model, instead serving as an "agent runtime and message router." It grants AI system permissions to operate the local computer directly and run autonomously, connecting commonly used messaging applications (such as WhatsApp, Discord, Telegram, and Signal) to backend LLMs (such as Claude, ChatGPT, or local models deployed through LM Studio/Ollama). Users can therefore remotely command AI through messaging applications to autonomously operate programs and complete tasks on their computers.

(I) Core architecture

1. Local gateway: a long-running background service based on Node.js listens for API requests from chat platforms and converts them into an instruction format that LLMs can understand.

2. Persistent memory: stores conversation history, user preferences, and tool outputs as local Markdown documents, making manual editing and review convenient and allowing efficient context accumulation through file reading.

3. Proactive heartbeat mechanism: unlike traditional question-answering chatbots, heartbeats let agents wake on a schedule in the background and autonomously advance long-running tasks without a user trigger.

4. Full-system access: exposing operating-system APIs or running Chrome extensions lets the model read and write local files, execute shell scripts, and automatically fill web forms. Simply put, it has all the permissions the user has.

(II) Workflow

When a user sends a natural-language instruction through a messaging application, such as "analyze bugs in my local codebase and generate a report," OpenClaw's underlying scheduling workflow is:

1. Input interception and parsing: the message router receives the user's natural-language input.

2. Context assembly: the system reads historical state, system prompts, and currently installed and available tools from local Markdown files.

3. Model reasoning: the assembled prompt is sent to the configured backend LLM, which outputs a tool-call action based on the current state.

4. Tool execution: the gateway intercepts the model's tool-call instruction (such as executing a terminal command or Python script), executes it in the local environment, and captures the result and environmental feedback. For OpenClaw's built-in tools, such as web\_fetch, OpenClaw directly executes its own codebase logic locally without calling through MCP (MCP is needed when the model calls external tools).

5. State update and feedback loop: the system writes execution results to persistent memory and updates the historical context. OpenClaw then determines whether the task is fully complete. If not, or if tool execution fails, the relevant information is fed to the model again for self-correction and another reasoning round. If the task is complete, a final textual response is returned through the messaging application.

(III) Security risks

OpenClaw gives agents extremely high local permissions, providing unprecedented convenience but also creating security risks. The community therefore recommends running it on a separate physical device (such as a Mac mini) or in an isolated environment.

## IV. DeepSeek Harness

Whereas Cursor, Claude Code, and Codex primarily start from "a directly usable coding agent," DeepSeek Harness emphasizes self-evolving infrastructure for building, running, and researching agents. The model supplies intelligence and reasoning, while a plugin-composed harness provides tools, environments, sessions, storage, sandboxes, loops, and scheduling, enabling sustained work in a computer environment.

(I) Main features

"Everything is a Plugin": models, tools, skills, sessions, sandboxes, storage, agent loops, schedulers, and even UIs are not hard-coded components; they can be replaced and recombined through configuration. Developers can replace the model, sandbox, tools, or entire agent loop while fully observing what the model saw and did at each step. This makes it suitable not only as a coding agent but also as a foundation for researching agent architectures and implementing self-evolving agent frameworks.

(II) Core architecture

1. Cordis plugin kernel: DeepSeek Harness is built on the Cordis plugin framework. Cordis manages plugins, service dependencies, events, and plugin loading and unloading, without prescribing specific agent functions. Model adapters, tool registries, session logs, and even the agent loop itself are plugins and can theoretically all be replaced.

2. Event-sourced sessions: the harness records events during an agent run in an append-only log that is never overwritten. User inputs, model outputs, tool calls, tool results, context injections, and subagent activities can all become events. This log is the session's "single source of truth," and the conversation history shown to the model is reconstructed from it.

3. Traceable trajectories: because execution is fully recorded as an event stream, developers can inspect what the model actually saw, why it called a tool, and what the tool returned. Sessions support resume, fork, search, and replay, making them particularly useful for debugging and studying agent behavior.

4. Default agent loop: it receives user input, creates a new turn in the session, combines the system prompt and history, and calls the model. If the model returns a tool call, the tool registry executes it and writes the result back to the session before the next model call.

5. Replaceable capability layer: filesystem, shell, web, LSP, subagents, sandbox, model adapters, session persistence, and storage are designed as independent capabilities. For example, session persistence can use JSONL, SQLite, or other backends without modifying the agent loop.

## References

- Anthropic. (2026). [Trustworthy agents in practice](https://www.anthropic.com/research/trustworthy-agents). Anthropic Research.
- Bonamy, N., & Choi, D. (2026). [Codex as a platform: build on the open agent harness](https://developers.openai.com/blog/codex-as-a-platform). OpenAI Developers.
- Cursor. (n.d.). [Cursor Agent: Overview](https://cursor.com/docs/agent/overview). Cursor Documentation. Accessed 2026-09-02.
- Cursor. (n.d.). [Cloud Agents](https://cursor.com/docs/cloud-agent). Cursor Documentation. Accessed 2026-09-02.
- Cursor. (n.d.). [Subagents](https://cursor.com/docs/subagents). Cursor Documentation. Accessed 2026-09-02.
- Cursor Team. (2026). [Cursor is now a part of SpaceX](https://cursor.com/blog/joining-spacex). Cursor.
- Anthropic. (n.d.). [Quickstart](https://code.claude.com/docs/en/quickstart). Claude Code Documentation. Accessed 2026-09-02.
- Anthropic. (n.d.). [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works). Claude Code Documentation. Accessed 2026-09-02.
- Anthropic. (2025). [Beyond permission prompts: making Claude Code more secure and autonomous with sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing). Anthropic Engineering.
- Bolin, M. (2026). [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/). OpenAI.
- Chen, C. (2026). [Unlocking the Codex harness: how we built the App Server](https://openai.com/index/unlocking-the-codex-harness/). OpenAI.
- OpenClaw. (n.d.). [OpenClaw Documentation](https://docs.openclaw.ai/). Accessed 2026-09-02.
- DeepSeek-AI. (n.d.). [DeepSeek Harness developer preview: Everything is a plugin](https://github.com/deepseek-ai/deepseek-harness). GitHub repository. Accessed 2026-09-02.
- Shi, Y., Zhang, W., & Cui, T. (2026). [A Programming Paradigm for Spatiotemporal Composability](https://arxiv.org/abs/2608.25512). arXiv:2608.25512.
