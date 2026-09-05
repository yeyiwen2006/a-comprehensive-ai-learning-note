---
title: "22.3 Multi-Agent Systems and Their Training"
chapter_title: "Agents"
section_id: "22-03"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/22.智能体/22.3 多智能体及其训练.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "all Word images manually transcribed as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 22.3 Multi-Agent Systems and Their Training

## I. Background and Advantages

1. Limitations of a single agent: a limited context window; limitations in reasoning depth and complex processing; opaque internal multistep processes and weak interpretability; insufficient specialized domain knowledge; susceptibility to hallucinations and weak self-correction; knowledge "frozen" at the training cutoff.

2. Advantages of multi-agent systems: avoiding excessively long contexts (for example, sending code execution results to another agent first to control the coding agent's context length); specialized division of labor, analogous to collaboration in human society; greater resilience, with limited impact from a single-node failure; parallel processing; flexibility; mutual correction to reduce hallucinations; and transparent inter-agent interactions with greater interpretability.

## II. Common Patterns of Multi-Agent Collaboration

1. Sequential: decompose multistep reasoning tasks, for example, assigning separate agents to problem formulation, experimental design, data analysis, and paper writing.

2. Role-based: assume different expert roles, analyze from different perspectives, and negotiate and collaborate, for example, as product managers and engineers.

3. Parallel: for example, partition a large dataset for different agents to process separately.

4. Debate-based: opposing agents debate, identify weaknesses, and challenge each other, with separate agents judging and checking facts. This is also an important way to avoid misleading outputs caused by hallucinations.

5. Fusion-based: different agents analyze the same question from different perspectives, then combine answers through simple voting, voting weighted by past accuracy, or output fusion, reducing the biases and errors of a single model.

6. External-interaction-based: specialized agents interact with different tools, databases, and external environments, for example, handling information retrieval, mathematical calculation, or document writing.

7. Plan–execute–verify: one agent formulates the plan, another executes it, and another independently verifies that the output meets requirements, avoiding having "the player also serve as the referee." When error tolerance is low, multiple sequential verification agents can be introduced.

These patterns can also be combined as needed.

## III. Typical Multi-Agent Architectures

1. Network: every agent can communicate with every other agent.

2. Supervisor: every agent communicates only with a single supervisor agent.

3. Hierarchical: an extension of the supervisor architecture, in which each agent communicates with its "immediate superior."

## IV. Are Multiple Agents Always Better Than One?

Without sound context engineering, blindly using a "parallel multi-agent architecture" can have disastrous consequences. Consider a Flappy Bird development example:

Idea: to speed up the work, let the agent split the task between subagents 1 and 2 for parallel execution.

Subagent 1 handles the background: unaware of what the other agent is doing, it creates a Super Mario-style background.

Subagent 2 handles the bird: it creates a bird unlike the one in the game, with the wrong movement.

Result: the two outputs cannot be combined.

Attempted fix: to address the disconnect, copy the original task into every subagent's context.

Still unsuccessful: real tasks contain substantial "tacit knowledge" generated through conversation and tool calls (for example, the user says midway, "make the bird red"). Subagents still misunderstand the task if they cannot see these dynamically generated intermediate states.

Final bottleneck (context overflow): trying to give every subagent everything that has happened causes token consumption to explode instantly on complex tasks, overflowing the context window.

In short, tasks such as code generation require exceptionally high logical coherence (tight coupling). Earlier variable definitions affect later function calls, so contexts must remain highly consistent. Splitting tasks as if they were reading-comprehension exercises produces code that does not run. Whether multiple agents outperform one depends on the task; the architecture serves the task. If multiple agents must be used, better methods of coordinating their contexts need to be explored.

## V. Multi-Agent Solutions to LLM Hallucinations

Reducing hallucinations through multi-agent collaboration, mutual evaluation, and debate is a common technical direction in current large-model research.

1. Mathematical principles

Suppose there are $n$ mutually independent models (agents), each with probability $p$ of making a "basic error" on the same question, where $0<p<0.5$, meaning it is more likely to be correct than wrong.

Under a "majority obeys the minority" or "unanimous approval" mechanism, changes in the error rate can be derived. Scenario A requires all models to agree before output (an extremely strict check). The probability $P_{\mathrm{error}}$ that multiple models simultaneously make the same specific error is:

$$
P_{\mathrm{error}}=\prod_{i=1}^{n}P(E_i)=p^n
$$

As $n$ increases, because $p<1$, $p^n$ approaches 0 exponentially.

Scenario B uses majority voting. With $n$ models ($n$ odd), if more than half ($k>n/2$) give the correct answer, the system outputs the correct result. By the binomial distribution, the probability $P_{\mathrm{correct\_sys}}$ that the system is ultimately correct is:

$$
P_{\mathrm{correct\_sys}}=\sum_{k=\lfloor n/2\rfloor+1}^{n}\binom{n}{k}(1-p)^k p^{n-k}
$$

By Condorcet's theorem, if individual accuracy $(1-p)>0.5$, then as $n\to\infty$, overall system accuracy $P_{\mathrm{correct\_sys}}\to 1$.

2. Real-world challenges: independence often does not hold

(1) Shared origins of training data create strongly correlated model outputs.

(2) RLHF makes models inclined to agree with other models' answers.

(3) For some logical traps or highly specialized knowledge, verification may be harder than generation.

3. Current technical approaches

(1) Chain-of-thought self-consistency.

(2) Majority voting after generation by different agents.

(3) Mutual evaluation using models with entirely different architectures or emphases.

(4) Adversarial debate between agents assuming different roles.

(5) Verification through external tools, such as code interpreters.

Random errors (basic calculation errors or slips) can be effectively addressed through (1), (2), and (3); systematic errors (knowledge gaps and logical traps) require (4) and (5).

## VI. Parallel-Agent Reinforcement Learning (PARL)

1. Core architecture

Taking Kimi K2.5 Agent Swarm as an example, when a task arrives during inference, an orchestrator determines the required number and roles of subagents (provided to them as context), assigns tasks, and schedules agents. Each subagent is a "clone" of the same large model with identical weights.

2. Training method

First train the subagents (the large model). Once trained, freeze their parameters and train the orchestrator with end-to-end reinforcement learning, initially using small models as subagents before replacing them with large models.

3. Reward function

The training signal consists of three parts, optimizing the orchestrator's parallel scheduling policy through RL:

$$
r_{\mathrm{PARL}}(x,y)=\lambda_1\cdot\underbrace{r_{\mathrm{parallel}}}_{\text{instantiation reward}}+\lambda_2\cdot\underbrace{r_{\mathrm{finish}}}_{\text{completion-rate reward}}+\underbrace{r_{\mathrm{perf}}(x,y)}_{\text{task-outcome reward}}
$$

(1) Parallel reward r\_parallel: mitigating "serial collapse"

Problem: the orchestrator may settle at a local optimum and default to serial execution with one agent.

Mechanism: reward subagent instantiation to encourage exploration of concurrent scheduling.

Role: force the system to attempt parallel decomposition rather than sequential execution.

(2) Completion reward r\_finish: preventing "fake parallelism"

Problem: the orchestrator may spawn many meaningless subagents, increasing parallelism metrics without actually decomposing the task.

Mechanism: reward only successfully completed subtasks, such as successfully returning query results, producing a final answer, or parsing a webpage.

Role: ensure feasible task decomposition and guide the policy toward effective parallel structures.

(3) Performance reward r\_perf: ensuring final quality

A verifiable reward based on the final task outcome, such as answer correctness.

As training proceeds, λ1 and λ2 are gradually annealed to 0, so that the policy ultimately optimizes the primary objective.

For tasks such as web-design aesthetics and open-ended generation that cannot be verified by rules or binary correctness, generative reward models (GRMs) can provide fine-grained evaluation. There may be one GRM (using different prompts for different tasks) or several. Prompts are designed to produce comprehensive, multidimensional, continuous scores. The final score is a weighted combination of verifiable rewards (such as whether retrieved facts are correct) and GRM scores, avoiding excessive reward hacking as much as possible.

4. Task design

PARL training tasks are specifically designed to stress-test the limits of sequential execution, inducing parallel behavior through synthetic prompts.

Broad search: explore multiple independent information sources simultaneously, requiring parallel access to multiple sources.

Deep search: multiple reasoning branches with delayed aggregation, allowing parallel exploration of different reasoning paths.

Long-document analysis: understand more than 100 input documents, processing different document segments in parallel.

Large-scale file downloading: acquire diverse resources in batches, requiring concurrent downloads.

5. Asynchronous coroutine architecture

Each agent task runs as an independent asynchronous coroutine, supporting recursive subtask calls (subagents can create further subagents).

## References

- 温睦宁、林江浩、张伟楠、俞勇. (2025). [*Hands-on Learning of Large-Model Agents*](https://haa.boyuai.com/) (translated title; in Chinese). Posts & Telecom Press. ISBN 978-7-115-68638-1.
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325). arXiv:2305.14325.
- Wang, X., Wei, J., Schuurmans, D., et al. (2023). [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://openreview.net/forum?id=1PL1NIMMrw). ICLR 2023.
- Kimi Team. (2026). [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276). arXiv:2602.02276.
