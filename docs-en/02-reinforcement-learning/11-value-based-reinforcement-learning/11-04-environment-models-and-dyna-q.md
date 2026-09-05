---
title: "11.4 Environment Models and Dyna-Q"
chapter_title: "Value-Based Reinforcement Learning"
section_id: "11-04"
language: en
source_language: zh
source_docx: "第2部分 强化学习/11.基于价值的强化学习/11.4 环境模型与Dyna-Q算法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 11.4 Environment Models and Dyna-Q

## I. Problems with Q-Learning Without an Environment Model

1. Delayed backward propagation

Standard model-free Q-Learning updates values as:

$$
Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma\max_{a^{\prime}}Q(S^{\prime},a^{\prime})-Q(S,A)]
$$

Note that this update spans only one step.

Consider a maze treasure hunt with path $S_0\to S_1\to S_2\to\cdots\to S_{99}\to$ treasure (+100).

1. First attempt (Episode 1): the agent wanders randomly and finally reaches the treasure.
   - Only the final step $S_{99}\to$ treasure receives a reward.
   - Result: only $Q(S_{99},\text{action})$ is updated and learns that something good is there.
   - Problem: $Q$ values at $S_{98}$, $S_{50}$, and even starting state $S_0$ remain 0. Before $S_{99}$, the agent has no idea it is on the right path.
2. Second attempt (Episode 2): the agent starts again.
   - It must randomly reach $S_{99}$ again.
   - Only when it moves from $S_{98}$ to $S_{99}$ and sees the high $Q$ value of $S_{99}$ does $Q(S_{98},\text{action})$ update.
   - Problem: propagating value from the end to the beginning requires 100 full episodes. This is painfully slow.

Delayed backward propagation severely slows updates and also makes the agent slow to respond to environmental changes. If the environment changes slightly (for example, a previous route becomes blocked), it often struggles to adjust its policy quickly and repeatedly fails in the new environment.

2. Poor sample efficiency

Accurate Q values require many interactions and samples. Collecting large numbers of samples is often expensive: every step in robot control consumes power, wears mechanical parts, and may cause damage; every trial-and-error trade costs real money. A computation, by comparison, is extremely inexpensive.

## II. Dyna-Q: Environment-Model-Based (Model-Based) Reinforcement Learning

1. Steps

A complete Dyna-Q iteration has four key steps.

**Step 1: interact with the environment**

In state $S$, the agent selects action $A$ from the current $Q$ table (usually with $\epsilon$-greedy), executes it, and receives reward $R$ and new state $S'$.

$$
(S,A)\xrightarrow{\text{Environment}}(R,S')
$$

**Step 2: direct reinforcement learning**

Use the real experience $(S,A,R,S')$ to update $Q$ with the Q-Learning rule:

$$
Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma\max_{a^{\prime}}Q(S^{\prime},a^{\prime})-Q(S,A)]
$$

This ensures the algorithm has the unbiasedness of model-free algorithms.

**Step 3: model learning**

Update the environment model using real experience. In a deterministic environment, simply record the result:

$$
\mathrm{Model}(S,A)\leftarrow(R,S^{\prime})
$$

This means that when the agent encounters $S$ and tries $A$ in its “imagination,” the model reports outcomes $R$ and $S'$.

An environment model (world model) is a function taking $s_t$ and $a_t$ and outputting $r_t$ and $s_{t+1}$. It can be a neural network.

**Step 4: planning**

This is the central distinction between Dyna-Q and Q-Learning. The agent uses the model for $N$ simulated updates:

Repeat $N$ times:

1. Randomly select state $\tilde{S}$ from previously observed states.
2. Randomly select action $\tilde{A}$ from actions previously taken at $\tilde{S}$.
3. Model prediction: input $(\tilde{S},\tilde{A})$ into the model to obtain predicted reward $\tilde{R}$ and next state $\tilde{S}'$:

$$
(\tilde{R},\tilde{S}^{\prime})\leftarrow \mathrm{Model}(\tilde{S},\tilde{A})
$$

4. Simulated update: update $Q$ again with Q-Learning, now using simulated data:

$$
Q(\tilde{S},\tilde{A})\leftarrow Q(\tilde{S},\tilde{A})+\alpha[\tilde{R}+\gamma\max_{a^{\prime}}Q(\tilde{S}^{\prime},a^{\prime})-Q(\tilde{S},\tilde{A})]
$$

Advanced environment models need not select only previously visited $(s_t,a_t)$ pairs and can choose more randomly. Having learned environmental regularities, the model often generalizes and can output a prediction.

(2) Further explanation

Suppose an agent receives a reward for reaching a goal in a maze.

Q-Learning: because the agent uses a Markov decision model, reward information propagates backward one step through Q values only when it approaches the goal again.

Dyna-Q: even near the start, the agent can revisit previous memories “mentally” through N planning steps. If it has experienced a complete start-to-goal path, planning uses model-generated simulated transitions to rapidly propagate high terminal values toward the start.

This resembles solving problems (the real environment): after solving one, we recall (model prediction) similar problems solved earlier (simulated experience), deepening understanding without redoing every problem.

(3) Dyna-Q's problem: model bias

Dyna-Q strongly depends on model accuracy. If Model(s, a) predictions disagree with reality (for example, the environment changes but the model has not been updated), planning propagates incorrect knowledge, producing policies that appear optimal in the model but fail in reality.

To address model failure from environmental changes, later work introduced Dyna-Q+, adding an exploration bonus to state-action pairs not visited for a long time:

Formula adjustment: during planning, simulated reward $\tilde{R}$ becomes $\tilde{R}+\kappa\sqrt{\tau}$, where $\tau$ is the number of time steps since the pair was last visited, and $\kappa$ is a coefficient.

This encourages checking long-unvisited places, updating the model promptly and correcting bias.

## References

- Sutton, R. S. (1990). [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](http://incompleteideas.net/papers/sutton-90.pdf). ICML 1990.
