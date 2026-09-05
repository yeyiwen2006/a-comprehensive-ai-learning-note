---
title: "25.2 Model Memory"
chapter_title: "Context and Memory"
section_id: "25-02"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/25.上下文与记忆/25.2 模型的记忆.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "all Word images manually transcribed as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 25.2 Model Memory

## I. Short-Term Memory (Context)

Short-term memory is the current task context available to a large model when performing a task: the messages, tool results, and other context tokens provided to the model. Inference systems usually use a KV cache to retain the key/value states of previously processed tokens at each attention layer, avoiding repeated computation during autoregressive generation. The KV cache is an inference optimization and runtime state, not the context itself; disabling it does not remove contextual information but increases repeated computation.

Because hardware GPU memory is limited, large models have finite context windows and must use certain mechanisms when the limit is exceeded. Measures for short-term memory itself include:

1. Sliding window: retain the most recent K rounds of conversation.

2. Summarization: summarize earlier content and "forget" the original, usually in conjunction with a sliding window. The summary is temporarily stored in memory and added to the context window of subsequent conversations.

Other methods combining short-term and long-term memory are discussed below.

## II. Long-Term Memory

We can embed memories as semantic vectors in a vector database and store them on high-capacity media such as hard drives. For example, OpenClaw stores memories in particular Markdown files in the workspace. When using them, retrieve the most relevant content from the database by vector similarity based on the request, or prompt the model to access it flexibly in other ways. Memories can include past questions and answers and their summaries, domain-specific knowledge, action histories and lessons learned, and user preferences.

When context exceeds the short-term memory limit, we can move the earliest historical information, or only its summary, to long-term memory. If it is retrieved later, meaning that this information is needed, it is added back to short-term memory.

Because long-term memory is slow to read and write, we generally do not access it in every conversation. Instead, we access it as needed during initialization (before starting work), retrieval (when certain problems or situations lead the LLM to judge that memory is needed), or archiving (when packaging and closing a completed project).

Long-term memory can be hierarchical. For example, OpenClaw includes a large database and a curated set of the most important memories, such as user preferences and major decisions; the curated memories are retrieved whenever interacting with the user. Claude-mem uses "progressive disclosure": after retrieving several memory entries through dense or sparse retrieval, it does not immediately inject all of them into context but processes them in layers. Specifically, memories are organized into the following three layers when created, allowing layered reading when accessed:

Layer 1: Search index layer: a "lightweight index" whose rows contain each memory's ID, time, category, title, and token cost. For example: \| \#2102\| Oct 20 \| problem-solution \| Fixed timeout in CI \| 100 tokens \|. Here, gotcha indicates a pitfall, while problem-solution indicates a past problem and its solution. Inject the index layer of retrieved entries into the main model. From titles and types, the model can decide whether an entry warrants the more expensive Layer 2 / Layer 3. (Unlike the preceding retrieval, which uses a fixed algorithm and is therefore relatively coarse and unable to handle fine-grained semantics, the model here can judge flexibly according to its needs despite receiving only a small amount of condensed information.)

Layer 2: Timeline layer: event memories, similar to a log. After selecting IDs in Layer 1, the model retrieves several timeline entries before and after each ID to see what happened around that operation.

Layer 3: Get Observations detail layer: if, after reading Layer 2, the model still considers more information necessary for an ID selected in Layer 1, it retrieves that ID's complete structured content. For example:

\#2543 Hook timeout: 60s too short for npm install

─────────────────────────────────────────────────

Date: Oct 26, 2025 2:14 PM

Type: gotcha

Project: claude-mem

Narrative:

Discovered that the default 60-second hook timeout is insufficient

for npm install operations, especially with large dependency trees

or slow network conditions. This causes SessionStart hook to fail

silently, preventing context injection.

Facts:

\- Default timeout: 60 seconds

\- npm install with cold cache: \~90 seconds

\- Configured timeout: 120 seconds in plugin/hooks/hooks.json:25

Files Modified:

\- plugin/hooks/hooks.json

Concepts: hooks, timeout, npm, configuration

In summary, this structured information includes:

Narrative:

Describe "what happened, what the problem was, and what its effects were" in natural language, helping the agent understand the context.

Facts:

Key numerical values, configuration items, and conclusions that facilitate logical reasoning.

Files Modified / Files Read:

Tell the agent directly which files an observation concerns, so it does not have to guess.

Concepts:

Tags for semantic retrieval, such as "hooks / timeout / npm," facilitating later concept-based retrieval.

In the official example, a typical pattern is: Layer 1 returns an index with limit=10; Layer 2 retrieves timelines for 2–3 selected IDs; Layer 3 retrieves complete observations for those that need them. Consider the following example:

Layer 1: call search({ query: "hook timeout", limit: 10 })

Obtain an index of 3 relevant memories:

\#2543: Hook timeout: 60s too short

\#2891: Hook timeout configuration

\#2102: Fixed timeout in CI

Layer 2: Claude judges \#2543 most relevant and calls timeline({ anchor: 2543, depth\_before: 3, depth\_after: 3 }), seeing:

Before: which timeout configurations were tried

Current: changing 60s to 120s

After: applying the same configuration in CI

Layer 3: if more detail appears necessary, call get\_observations({ ids: [2543, 2102] })

Obtain the full narrative, facts, modified-file list, and other information for the current decision.

## III. RAM

During an AI agent's operation, system RAM / CPU memory acts as a "dispatcher" and "transfer station," bridging the hard drive (database) and GPU (LLM). The agent program we write runs in RAM.

Some information exceeds short-term memory capacity but is accessed too frequently for long-term memory's retrieval efficiency to be acceptable. We can therefore create data structures in the program and store this information in RAM, as in the following two examples.

1. Retaining historical information by importance

Establish a scoring mechanism, such as TF-IDF, attention weights, or a scoring model, to assess the importance of historical information. Retain the K most important items, such as K=5 or 10. This suits tasks whose key information is spread across a long period.

For the data structure, we can implement a dynamically changing heap. Each new conversation entry receives an importance score and is added according to that score. The heap retains only the K highest-scoring entries. When entry K+1 arrives, discard it if its score is too low; if it scores higher than the lowest entry in the heap, remove that lowest entry and insert the new one. Before the next question and answer, add the heap's contents to short-term memory, that is, the context. The following example explains why this process must take place in a heap rather than in the context:

Your context window can hold only 3 memories.

Without a heap (directly entering the context):

1. The user says, "My name is Xiaoming." $\to$ store it in context (2 spaces remain).
2. The user says, "The weather is nice today." $\to$ store it in context (1 space remains).
3. The user says, "Have an apple." $\to$ store it in context (full).

Unexpected event: the user suddenly says, "I am allergic to penicillin!"

- The context is already filled with the preceding 3 pieces of chatter. To add this most important item, you need complicated logic to "erase" an existing entry. Performing "find and replace" in the prompt string is cumbersome and inefficient.

- **Operating in the heap**:
  - Object: a Python list/object.
  - Cost: nanoseconds of CPU time, zero monetary cost.
  - Flexibility: freely add, delete, change weights, and reorder entries.

- **Operating in the context**:
  - Object: the prompt string sent to the API.
  - Cost: tokens are billed by usage. Once the prompt is sent, it cannot be taken back.

The heap is scratch paper; the context is the final exam paper.

- On scratch paper, you can repeatedly calculate, revise, and compare to decide which questions matter most (sorting and replacing entries in the heap by score).
- Once you write the answers on the exam paper (put them in the context window) and submit it (send it to the LLM), you cannot change them.

2. Generating a summary

As mentioned earlier, when a conversation exceeds the context limit, let the LLM summarize the preceding conversation and inject the summary into the subsequent context window. In program form, this is something like summary=agent.ask(massage), where massage contains the prompt requesting a summary. Here, summary is temporarily held in RAM, awaiting injection into the new context.

A summary does more than shorten long text, which would lose many details. It should preserve, as far as possible, the goals, constraints, completed work, and unresolved issues needed to continue the task. When context tokens exceed a threshold, OpenAI Codex calls the Responses API's `/responses/compact` endpoint, replacing the old input with a smaller set of input items representing the original conversation. These include a `type=compaction` item with encrypted content to retain the model's latent understanding of the conversation. Earlier Codex versions replaced old input with summary messages, but the current implementation is not a fixed "small project list."

## References

- 温睦宁、林江浩、张伟楠、俞勇. (2025). [*Hands-on Learning of Large-Model Agents*](https://haa.boyuai.com/) (translated title; in Chinese). Posts & Telecom Press. ISBN 978-7-115-68638-1.
- OpenClaw. (n.d.). [OpenClaw GitHub repository](https://github.com/openclaw/openclaw). Accessed 2026-09-02.
- thedotmack. (n.d.). [claude-mem GitHub repository](https://github.com/thedotmack/claude-mem). Accessed 2026-09-02.
- Hugging Face. (n.d.). [Cache strategies](https://huggingface.co/docs/transformers/main/kv_cache). Transformers Documentation. Accessed 2026-09-02.
- Bolin, M. (2026). [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/). OpenAI.
