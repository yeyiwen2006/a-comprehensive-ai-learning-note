---
title: "24.3 Skills"
chapter_title: "Tool Use"
section_id: "24-03"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/24.工具调用/24.3 Skills.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "not applicable; source contains no Word images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 24.3 Skills

## I. The Basic Idea

An agent's skills can be understood as its "specialized capability packages": reusable local instructions, workflows, and references, with scripts, templates, or tool constraints when necessary. By invoking an appropriate skill for a task, the agent need not reason from scratch every time and can follow an established, stable workflow.

Core value:

1. Preserve experience, such as fixed steps, common pitfalls, and validation methods, ensuring consistent repeated execution.

2. Reduce errors: skills often explicitly specify which files to read, what order to follow, and when to stop for confirmation, making them more reliable than improvisation.

3. Improve efficiency: for complex tasks, agents need not reorganize the entire workflow. They apply an appropriate skill and spend time on the parts genuinely requiring judgment and creativity.

A good skill explains not only "what to do" but also "why to do it this way, how far to go, and how to verify the results." Agent capabilities become explainable, inspectable, and continually improvable rather than vague.

## II. Encapsulating Tool Interfaces with Skills

Storing tool information and tool results directly in a model's context window fills it and reduces efficiency. Moreover, model tool calls through MCP sometimes still produce mismatched output formats.

Skills can encapsulate tool interfaces. Specifically, the model calls tools by running code packaged in skills, adding scripts itself when needed. Tool results are not injected directly into context but processed in a sandbox by the packaged code or model-written code.

This uses the model's strong programming abilities to improve success rates and lets it control data indirectly through code rather than directly, avoiding unnecessary injection of large numbers of tokens into context and improving efficiency.

## References

- Anthropic. (n.d.). [Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview). Claude API Docs. Accessed 2026-09-02.
