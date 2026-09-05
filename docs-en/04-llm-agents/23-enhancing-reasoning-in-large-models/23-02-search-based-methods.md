---
title: "23.2 Search-Based Methods"
chapter_title: "Enhancing Reasoning in Large Models"
section_id: "23-02"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/23.大模型的推理增强/23.2 基于搜索的方法.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "all Word images manually transcribed as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 23.2 Search-Based Methods

Earlier, we noted that an LLM can only output the probabilities of choosing each token, regardless of how it is trained, including reinforcement learning. "Choose the highest-probability token at each step" is a one-step greedy operation that may miss a global optimum (for example, probabilities 0.6 at step 1 and 0.1 at step 2 versus 0.4 at step 1 and 0.9 at step 2). This motivated sampling methods such as top-k sampling, temperature adjustment, and beam search, which retains the k best paths at each step so it can backtrack when a path becomes unpromising. But these methods do not solve the fundamental problems of "considering only one token's probability" and "following only one path." Humans reason about difficult problems in steps, not tokens, often trying several paths for a step before choosing.

To overcome the tendency of a single reasoning path to become trapped in local optima or fail at one point, we draw on beam search and introduce search-based reasoning enhancement. It lets agents explore multiple paths through the solution space to obtain a globally optimal path as closely as possible. Mathematically, linear and search-based reasoning can be expressed respectively as:

$$
\hat{y}=\arg\max_y P(y\mid x)
$$

$$
\hat{y}=\arg\max_y\sum_{z\in\mathcal{Z}}P(y\mid z,x)P(z\mid x)
$$

Here, y denotes the generated sequence and z denotes the collection of reasoning paths explored in parallel.

The expressions are mathematically equal but have very different practical meanings. The first is end-to-end. If the model has not learned well or the intermediate logic is too deep, directly fitting P(y\|x) is error-prone, and argmax can mislead (for example, correct approach z\_1 has probability 0.3, correct approach z\_2 has probability 0.3, and incorrect approach z- has probability 0.4; choosing the maximum sends the model racing in the wrong direction). In the second, p(z\|x) generates a reasonable solution approach from the problem, and p(y\|z,x) generates an answer from that approach. Explicitly modeling z makes it easier to "think clearly." This turns vague intuition in the model's head (hidden-layer vectors) into clear scratch work (tokens), and determining a probability distribution over z helps avoid the aforementioned randomness of argmax.

Also note that a search algorithm is a "sampling program" external to the model, just as humans often need to work through complete steps on scratch paper. The model itself, whether trained with RL or not, can only generate next-token probabilities. Ordinary sampling acts directly on these probabilities, whereas search-based methods first generate content in parallel along multiple paths, then choose the best as output.

Some common methods follow:

## I. Tree of Thoughts

Introduce a tree structure in the main program to formalize model reasoning as tree search.

1. Generation: generate multiple candidate thought branches at the current node (state). A branch represents a complete reasoning step, not a token; token-level branching would make the tree too wide, deep, and sparse for effective value evaluation. Prompts require a special token to mark the end of a step. Top-k sampling retains only a few branches, such as 5.

If a task must search with token-level branches, such as code completion, the model itself or a value network can first select the top-k tokens.

2. Evaluation: score the potential of each thought node using the model itself or an external function to decide whether to retain or prune it.

3. Search strategy: breadth-first search (BFS) or depth-first search (DFS).

## II. Self-Consistency

Reason independently about the same question multiple times, then use voting to output the most frequent answer. The principle is that incorrect reasoning usually has different causes and paths, making its outputs less likely to concentrate on one result. Correct reasoning paths may differ in form but often converge to the same answer, giving it an advantage in probability and sampled frequency.

## III. LATS: A Language-Agent Algorithm Incorporating Monte Carlo Tree Search

Monte Carlo tree search (MCTS) predates LATS and was used by AlphaZero for board-game decisions. LATS, proposed in 2023 and published at ICML 2024, applies tree search to language-model reasoning, acting, and planning. We first introduce the general mechanisms of MCTS and AlphaZero, then explain LATS's language-agent extensions. They should not be treated as the same algorithm.

1. Core principles (inference workflow)

(1) Stage 1: Explore

Starting at the root, select the most promising child to explore. An upper-confidence-bound (UCB) strategy balances the node's average value (the path's past success probability) against its visit count, avoiding neglect of less-explored paths. A common formula is:

$$
\mathrm{UCT}=\frac{w_i}{n_i}+C\sqrt{\frac{\ln N}{n_i}}
$$

Here, $w_i$ is cumulative node value, $n_i$ is node visit count, $N$ is parent visit count, and $C$ is an exploration constant.

In practice, an improved form, PUCT, is often used:

$$
a_t=\arg\max_a\left(Q(s,a)+U(s,a)\right)
$$

$$
U(s,a)=C_{\mathrm{puct}}\cdot P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
$$

Q(s,a) is the value function and P(s,a) is the policy function pi(s,a), provided respectively by trained value and policy networks that no longer update after training. In MCTS-based inference, pi(s,a) is not directly output as the actual policy. Instead, it serves as a prior, indicating "where intuition says to go," so that exploration initially favors promising moves rather than trying randomly. N(s,a) is the number of times action a is taken in state s.

(2) Stage 2: Expand paths

Upon encountering a leaf that is not the end of reasoning, generate multiple candidate operations from the current context and historical memory, such as the next solution step, a code fragment, or an action, not necessarily a single token. These candidates become new children in the search tree.

(3) Stage 3: Evaluate leaf values

The model continually executes generated actions along the current path to obtain feedback and identify the best path. If the task involves an external environment, such as programming, the system obtains compiler or environmental feedback. Each tree node needs a value score, which may come from:

An evaluation network: train a network, with weights frozen at inference, to fit "the expected final reward after generating this step." Traditional MCTS requires random rollout moves until the game ends. But LLM-generated long-text spaces are enormous and costly; for LLM training with a critic value network, such as PPO, the final token's value from the value network can directly approximate the evaluation score with little bias.

Model-based evaluation: the LLM scores itself, for example, by being asked, "Is this path logically coherent?"

Environmental feedback: hard metrics such as the proportion of test cases passed.

Self-reflection: LATS integrates "external interaction" and "long-term memory" into search. When a simulation fails, the model automatically generates a natural-language reflection summary and writes it to a memory pool. During subsequent expansion, the LLM reads these reflections. This means the agent does not fall into the same trap twice, instead using past failures to correct its current generation policy.

(4) Stage 4: Back up values to nodes along the path

When the current reasoning path finishes, returning toward the root updates each node's visit count and propagates execution results (rewards/values) upward along the path. Each node's average value is the mean reward of all leaves whose values were backed up to it (for example, in AlphaZero, leaf rewards are V=1 for a win, V=0 for a draw, and V=-1 for a loss):

For every node $(s_t,a_t)$ along the path, the exact updates are:

1. Update total action value (cumulative sum):

$$
W(s_t,a_t)\leftarrow W(s_t,a_t)+V_{\mathrm{leaf}}
$$

2. Update visit count:

$$
N(s_t,a_t)\leftarrow N(s_t,a_t)+1
$$

3. Calculate the average value used by UCB/PUCT in Stage 1:

$$
Q(s_t,a_t)=\frac{W(s_t,a_t)}{N(s_t,a_t)}
$$

This also means that when multiple explored paths pass through the same node, their arithmetic mean is taken. The tree can thus "remember" which branches lead to high returns and which are dead ends.

An important distinction:

At test time, neural-network parameters $\theta$ are frozen (fixed).

- **Unchanged prior probabilities**: for any given position $s$, the policy network's action probabilities $P(s,a)$ are completely unchanged.
- **Unchanged individual evaluations**: the value network's raw score for a given leaf, $v=f_\theta(s_{\mathrm{leaf}})$, is also completely unchanged.

However, $Q(s,a)$ in the MCTS tree is highly dynamic. $Q(s,a)$ is not a fixed number directly output by the network, but an empirical average continually revised as exploration deepens over hundreds or thousands of test-time simulations.

(5) Stage 5: Final decision

When the search budget, such as maximum node count or time, is exhausted, the algorithm must output a result. Usually, the most-visited root branch is selected as the best solution because MCTS theory regards high visit counts as indicating a path that has survived repeated evaluations and performed well. Alternatively, each node's share of visits can be used as its output probability for sampling. Key reflections are also added to long-term memory, accumulating experience for the next task.

2. AlphaZero

(1) Definition of the Markov state

To ensure the environment is Markovian, state s must include the black and white pieces and whose turn it is. During training, the same neural network controls both sides, with each step maximizing the current side's winning probability. In many board games, future state s\_t+1 and legal moves depend not only on the current board but also on past events, so several preceding board positions must also be included in the state. Take Go as an example:

(2) Expanding Monte Carlo tree search

The MCTS "tree" is essentially a partial expansion of the actual game tree. In two-player alternating-turn games, such as Go and chess, its levels correspond exactly to real alternating turns:

- **Root (depth $d=0$)**: the actual current board, with "me" to move.
- **Child (depth $d=1$)**: the board after "I" hypothetically make a move, with the "opponent" to move.
- **Grandchild (depth $d=2$)**: the board after the "opponent" also moves, with "me" to move again.

And so on. When MCTS descends to an unexpanded leaf $s_L$, it calls the network to evaluate it:

$$
(\mathbf{p},v)=f_\theta(s_L)
$$

The value $v\in[-1,1]$ is always a winning probability from the perspective of "the side to move at the current node $s_L$."

- If it is "my" turn at $s_L$, $v=0.8$ means "I" have an excellent chance of winning.
- If it is the "opponent's" turn at $s_L$, $v=0.8$ means the "opponent" has an excellent chance of winning (in other words, "I" am close to losing). Because parent and child nodes belong to different sides, each upward propagation must multiply $v$ by $-1$, flipping its sign. Suppose a leaf at depth 3, with the opponent to move, is evaluated as $v=+0.9$ (a major advantage for the opponent).

- Back up to depth 2 (my turn): record the action value as $-0.9$ (I am in danger).
- Back up to depth 1 (opponent's turn): record $-(-0.9)=+0.9$ (the opponent is safe).
- Back up to the root at depth 0 (my turn): record $-(+0.9)=-0.9$ (I must not choose the branch leading here). At every search depth, PUCT uses exactly the same maximization formula:

$$
a_t=\arg\max_a\left(Q(s_t,a)+c_{\mathrm{puct}}\mathbf{p}(a\mid s_t)\frac{\sqrt{\sum_b N(s_t,b)}}{1+N(s_t,a)}\right)
$$

- **The current node is my turn**: PUCT seeks the action maximizing "my" $Q$, while using $\mathbf{p}$ and $N$ to explore unknown possibilities.
- **The current node is the opponent's turn**: PUCT still maximizes $Q$, but $Q$ is now calculated from the opponent's perspective, effectively simulating that "the opponent will choose the action most beneficial to them and most dangerous to me."

Thus, MCTS considers not only "what should I do later after this move?" but also "what will the opponent do after this move?"

(3) RL training method

Selection and expansion: starting from root $s_t$, repeatedly use network $f_\theta$ to evaluate leaves and select actions through PUCT to search downward until reaching a leaf. Action selection considers both winning probability $Q(s,a)$ and exploration bound $U(s,a)$:

$$
a_t=\arg\max_a\left(Q(s,a)+c_{\mathrm{puct}}\mathbf{p}(a\mid s)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}\right)
$$

Output an improved policy: after hundreds or thousands of simulations, MCTS produces a stronger policy distribution $\pi_t$ from root-branch visit counts $N(s_t,a)$:

$$
\pi_t(a\mid s_t)=\frac{N(s_t,a)^{1/\tau}}{\sum_b N(s_t,b)^{1/\tau}}
$$

Here, $\tau$ is a temperature parameter controlling exploration.

Execute and record: sample action $a_t$ from $\pi_t$ and execute it on the actual board. Record the current state and MCTS-improved policy $(s_t,\pi_t)$.

Game end: the final actual outcome is $z\in\{-1,0,1\}$. Package the entire game's data into tuples $(s_t,\pi_t,z)$ and store them in a replay buffer. A separate parallel process randomly samples minibatches $(s,\pi,z)$ from the buffer and updates network parameters $\theta$ through gradient descent to minimize prediction error. The loss contains two parts, value mean squared error and policy cross-entropy, plus L2 regularization:

$$
\mathcal{L}(\theta)=(z-v)^2-\boldsymbol{\pi}^{T}\log\mathbf{p}+c\lVert\theta\rVert^2
$$

- $(z-v)^2$: forces the network's value estimate $v$ toward actual outcome $z$ (the true value approaching the Nash equilibrium).
- $-\boldsymbol{\pi}^{T}\log\mathbf{p}$: forces prior probabilities $\mathbf{p}$ toward the better policy $\boldsymbol{\pi}$ found by MCTS (policy improvement).

3. Applications in large models

(1) Truncated evaluation

Original MCTS reasons step by step to the end before backing up values. For long reasoning, this is extremely time-consuming and costly. Therefore, for each path, we score it with an evaluation network immediately after generating one new node layer, using truncated evaluation. Only when several paths have similar scores and the final output is highly uncertain do we generate another step.

(2) When is MCTS needed?

MCTS can be used for sampling during reinforcement fine-tuning and inference. Because this changes the actual actions sampled, the advantage function used for parameter updates must be adjusted to match the actual sampling policy. For example, in PPO, when calculating A(s,a), Q(s,a) is taken directly as the average value of all MCTS-sampled paths starting from s and passing through a. If generation is truncated after one step, a path's value equals the total actual return of all tokens in that step plus the final token's expected future return. This uses "value under the actual sampling policy" and has less bias than plain temporal differences that "look ahead only one token." V(s) remains the value network's output for input s. This is also the idea of expert iteration discussed later: A(s,a) increases for high-value paths found by search, improving the policy to increase action a's probability, but without directly imitating through supervised learning.

During forward inference, search slows generation, so this method, or other search-based methods such as tree of thoughts, is generally used only in slow-thinking modes and long reasoning problems, not in fast-thinking modes or everyday question answering. Even without explicit inference-time search, however, the model has learned this search pattern during training, producing token-by-token reasoning such as "try method A... discover it does not work... backtrack... try method B."

Explicit MCTS is computationally expensive because it maintains a search tree and repeatedly evaluates candidates. DeepSeek-R1 does not explicitly run MCTS at inference, instead using GRPO for reinforcement learning. First introduced in DeepSeekMath, GRPO is a PPO variant that estimates advantages from relative rewards within sample groups, avoiding a separate critic. With clear reward signals and sufficient sampling, pure RL may produce token-sequence reasoning patterns such as "try–check–correct," but this is not equivalent to running inference-time MCTS with node expansion and value backup.

## IV. Retrieval of Relevant Reasoning

Have the model recall approaches to related problems or useful subproblems while reasoning, much as someone solving a new problem recalls similar problems or intermediate results encountered before. This resembles goal-directed RL: although the model does not know how to reach A, it may have learned how to reach B and C and can infer a route to A through analogy and reasonable extrapolation. This also illustrates retrieval's importance in reasoning.

## References

- Yao, S., Yu, D., Zhao, J., et al. (2023). [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601). NeurIPS 2023.
- Wang, X., Wei, J., Schuurmans, D., et al. (2023). [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://openreview.net/forum?id=1PL1NIMMrw). ICLR 2023.
- Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404). Science.
- Shao, Z., Wang, P., Zhu, Q., et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv:2402.03300.
- DeepSeek-AI. (2025). [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). arXiv:2501.12948.
- Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., & Wang, Y.-X. (2023). [Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models](https://arxiv.org/abs/2310.04406). ICML 2024.
