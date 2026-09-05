---
title: "14.5 Goal-Oriented Reinforcement Learning"
chapter_title: "Other Reinforcement Learning Algorithms"
section_id: "14-05"
language: "en"
source_language: "zh"
source_docx: "第2部分 强化学习/14.强化学习的其他算法/14.5 目标导向的强化学习.docx"
status: "translated"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 14.5 Goal-Oriented Reinforcement Learning

## I. Goal-Oriented Reinforcement Learning: Making the Goal a Variable

### (I) Goal Orientation

In reinforcement learning, we sometimes want a model to transfer to general tasks rather than complete only one task.

The differences between traditional and goal-oriented reinforcement learning can be summarized as follows:

| Item | Traditional reinforcement learning | Goal-oriented reinforcement learning |
| --- | --- | --- |
| Task form | Learns only one fixed task | Takes goal $g$ as an input variable and learns a family of tasks |
| Value function | $Q(s,a)$ | $Q(s,a,g)$ |
| Policy | $\pi(a\mid s)$ or $\pi(s)$ | $\pi(a\mid s,g)$ or $\pi(s,g)$ |
| Reward | Mainly determined by states and actions | Depends on the distance between the current state and goal $g$, or whether the goal has been reached |

Mathematically, this adds a dimension g to the Markov decision process (MDP). Reward function R no longer considers only s, but the distance between s and g.

### (II) The Sparse-Reward Dilemma

In robotic-arm grasping or ball-pushing tasks, sparse rewards cause serious data waste. Most exploratory trajectories do not reach the exact specified goal and receive only failure rewards. Only a tiny fraction succeeds by chance and receives positive rewards.

| Trajectory result | Reward signal | Learning problem |
| --- | --- | --- |
| Repeated attempts fail to reach the specified goal | $r=-1$ or $r=0$ | The model knows only that it “did not succeed,” not where its action pushed the object |
| Reaches the goal by chance | $r=+1$ | Too few successful samples to support stable training |

### (III) Hindsight Experience Replay (HER)

The core idea of hindsight experience replay (HER) is this: if an agent intended to reach goal A but actually reached goal B, the experience is a failure for A but can be reinterpreted as an experience of “successfully reaching B.”

The original experience can be written as:

$$
\{s,a,r=-1,s',\mathrm{Goal}=A\}
$$

Relabeling the goal in hindsight gives:

$$
\{s,a,r=+1,s',\mathrm{Goal}=B\}
$$

Although the agent has not yet learned to reach A, it first learns to reach B. As it learns to reach more places (B on the right, C to the front-right, D on the left, E directly ahead...), it learns the movement patterns for reaching different points and can eventually learn to reach A. Consider another example, archery:

If you aim at bullseye A but hit nearby B, this is a failure from the perspective of goal A. From the perspective of “how to hit B,” however, the action provides valid experience. HER systematizes this hindsight reinterpretation: failed trajectories are no longer simply discarded, but relabeled as successful samples for another goal.

The agent can thus learn from failure.

The workflow follows:

HER's training procedure can be written as:

1. Sample an original trajectory:

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_T)
$$

2. Store ordinary transitions under original goal $g$:

$$
(s_t,a_t,r_t,s_{t+1},g)
$$

3. Select an actually reached future state $g'=s_k$ from the same trajectory as the new goal, recalculate the reward, and store the relabeled transition:

$$
(s_t,a_t,r(s_t,a_t,g'),s_{t+1},g')
$$

4. Store ordinary and relabeled experiences together in the replay buffer.
5. Train a goal-conditioned policy with an off-policy reinforcement learning algorithm such as DDPG, SAC, or DQN.

### (IV) Value Function

The data above mix rewards under different goals, so DDPG, SAC, or DQN cannot be applied directly. We therefore modify the value function into a universal value function approximator (UVFA).

The UVFA modification feeds the goal into both the value function and the policy network:

$$
Q(s,a)\quad\longrightarrow\quad Q(s,a,g)
$$

$$
\pi(s)\quad\longrightarrow\quad \pi(s,g)
$$

Thus g becomes one of the function's inputs, often represented as a vector (such as a position in three-dimensional space). Because neural networks generalize, even if the network has never reached a, it can make a generalized estimate from data for nearby goals. There are three reasons this estimation is possible:

1. Physical dynamics are shared. The goal changes, but the environment's state-transition rules do not: the same robotic arm, tabletop, and object still follow the same dynamics.
2. Goal spaces are usually continuous. If goals $g_1$ and $g_2$ are spatially close, the action sequences needed to reach them are usually similar, and the value-function outputs should be close.
3. Relative positions transfer. Many tasks actually depend on the relative vector:

$$
\Delta=g-s
$$

When two tasks have similar $\Delta$, their policies can reuse similar action patterns even if their starting and ending positions differ in absolute terms.

This generalization can also be understood through Bellman propagation: if the path from current position $s$ to intermediate point $B$ has been learned and the value from $B$ to goal $g$ can be estimated, goal $g$'s value information can propagate backward through $B$.

The goal-conditioned Bellman target is:

$$
y
=
r(s,a,g)
+
\gamma
\max_{a'}Q(s',a',g)
$$

With a deterministic policy, it can also be written as:

$$
a'=\pi(s',g)
$$

$$
y
=
r(s,a,g)
+
\gamma Q(s',\pi(s',g),g)
$$

Taking DDPG + HER as an example, the specific procedure is:

The goal must be concatenated to the network input:

| Network | Input | Output |
| --- | --- | --- |
| Actor | $\mathrm{concat}([s,g])$ | Action $a=\pi(s,g)$ |
| Critic | $\mathrm{concat}([s,a,g])$ | Goal-conditioned value $Q(s,a,g)$ |

Data for different goals can then be mixed in one replay buffer for training. The buffer may simultaneously contain goals $g_1,g_2,g_3,\ldots$, but because goal $g$ itself is part of the network input, the network learns a goal-conditioned mapping rather than collapsing all goals into the same label. In other words, the same state and action may have high value under goal $g_A$ and low value under goal $g_B$. The network uses input $g$ to distinguish these cases, so mixing data for different goals does not directly create a conflict.

The target networks and loss functions for DDPG + HER can be written as:

$$
a_i'=\pi'(s_i',g_i)
$$

$$
y_i
=
r(s_i,a_i,g_i)
+
\gamma Q'(s_i',a_i',g_i)
$$

The Critic loss is:

$$
L_Q
=
\frac{1}{N}
\sum_i
\bigl(
Q(s_i,a_i,g_i)-y_i
\bigr)^{2}
$$

The Actor's objective maximizes the goal-conditioned value evaluated by the Critic:

$$
J_\pi
=
\frac{1}{N}
\sum_i
Q(s_i,\pi(s_i,g_i),g_i)
$$

The network thus becomes a “universal navigator”: given a starting point s and destination g, it can use physical laws to tell you how to move, without being restricted to the fixed destination used during training.

### (V) Current Problems

Although goal-oriented reinforcement learning can learn from failure, when the action space is enormous, tasks are weakly related, and no “reward relevant to the main objective” can be found for a long time, it is still like trying to understand an elephant by touch while blind.

## II. Self-Supervised Goal-Oriented Reinforcement Learning (NeurIPS 2025)

Unlike methods such as SAC/PPO that are based on dynamic programming (Bellman equations), the method in this paper is essentially a self-supervised sequence-modeling problem.

### (I) Model Components

1. Inputs and outputs

The self-supervised goal-oriented method treats future states in the same trajectory as goals. Given current state $s_t$ and future state $g=s_{t+k}$, the model first encodes both:

$$
z_0=\mathrm{Embed}(s_t,g)
$$

A deep network then produces two output heads:

| Module | Input | Supervision signal |
| --- | --- | --- |
| Policy head $\pi_\theta$ | Current state $s_t$ and goal $g=s_{t+k}$ | Predict actual action $a_t$ |
| Value/distance head $V_\theta$ | Current state $s_t$ and goal $g=s_{t+k}$ | Predict steps $k$ needed to reach the goal |

Summary:

Training samples can be organized as:

| Input | Target output |
| --- | --- |
| $(s_t,s_{t+k})$ | Current action $a_t$ |
| $(s_t,s_{t+k})$ | Temporal distance $k$ |

2. Loss functions and updates

Action prediction uses negative log-likelihood loss:

$$
L_{\mathrm{action}}(\theta)
=
-
\mathbb{E}
\bigl[
\log \pi_\theta(a_t\mid s_t,g=s_{t+k})
\bigr]
$$

If distance is predicted, there is also a distance-prediction loss:

$$
L_{\mathrm{dist}}(\theta)
=
\mathbb{E}
\bigl[
\lVert
V_\theta(s_t,s_{t+k})-k
\rVert^{2}
\bigr]
$$

3. Network architecture

These methods emphasize very deep residual networks. An ordinary residual block can be written as:

$$
x_{l+1}=x_l+F(x_l)
$$

Deep gated ResNet blocks add spectral normalization, normalization layers, and gating strengths so that each layer makes only a controlled, small update:

$$
x_{l+1}
=
x_l
+
\alpha_l\cdot
\sigma
\bigl(
\mathrm{BN}
\bigl(
W_2\,
\phi
\bigl(
\mathrm{BN}(W_1x_l)
\bigr)
\bigr)
\bigr)
$$

### (II) Core Idea

The core idea treats network depth as “internal computation time.” Shallow networks can only make short local mappings, whereas networks with hundreds or thousands of layers can progressively refine intermediate representations during forward propagation, effectively performing multistep search inside the model.

Different depths can be understood as different degrees of reasoning:

| Layer | Intuition about capability |
| --- | --- |
| Layer 10 | Mainly local feature processing and short-range action prediction |
| Layer 500 | Can integrate longer-range trajectory information |
| Layer 900 | Closer to multistep planning or internal search |

### (III) Why Is Increasing Network Depth Unsuitable for Traditional RL Algorithms?

Ordinary RL does not naturally improve just because the network becomes deeper, mainly because its supervision signal is weak. Methods such as SAC and TD3 usually rely on scalar $Q$ values and bootstrapping targets that change with network updates. Deeper networks are more susceptible to accumulated errors and unstable targets during training.

Self-supervised goal-oriented methods provide denser signals: each trajectory yields many $(s_t,s_{t+k})$ pairs, with structured supervision such as action and distance prediction for each sample. Network capacity and training signals therefore match better.

| Method | Main supervision signal | Risk or benefit of greater depth |
| --- | --- | --- |
| Traditional RL such as SAC/TD3 | Scalar $Q$ values dependent on bootstrapping | Deep networks can amplify target drift and estimation errors |
| Self-supervised goal-oriented RL | Dense signals such as actions, distances, and goal-conditioned trajectory relationships | Depth can translate into longer internal search and compositional reasoning |

## III. Why Are Goal-Oriented Methods Still Mainly Restricted to Fields Such as Robotics?

Although GORL is very successful in robot control, large-scale application to large language models faces the following core challenges:

1. Unbounded goal spaces and semantic ambiguity

In robotics, goals are usually coordinates (x, y, z); in LLMs, goals are natural-language instructions, such as solving a problem. Semantic space is discrete and high-dimensional. It is difficult to devise an automated, non-hackable method that reframes a failed answer as a successful answer for another goal (both identifying and verifying that goal are difficult). Moreover, an LLM's state s already contains the context, implicitly including the goal. Forcing goal and process apart would only create trouble.

2. The difficulty of state representation

An LLM's state s is the generated token sequence. Defining “distance to the goal” in text generation is highly abstract. Current RLVR is outcome-based and considers only the endpoint, whereas GORL relies heavily on modeling the state-transition process.

3. Computational cost and data efficiency

GORL requires the model to keep changing goals for exploration during training. LLM inference is extremely expensive. Sufficient exploration in an enormous goal space requires exponentially growing computation. Current RLVR (such as GRPO) is already very resource-intensive with its within-group relative comparisons; GORL's multigoal sampling would make training costs even harder to sustain.

## References

- Schaul, T., Horgan, D., Gregor, K., & Silver, D. (2015). [Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html). ICML 2015.
- Andrychowicz, M., Wolski, F., Ray, A., et al. (2017). [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495). NeurIPS 2017.
- Wang, K., Javali, I., Bortkiewicz, M., Trzcinski, T., & Eysenbach, B. (2025). [1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e74ee34cc0f2d0780f34ee77d8fba25b-Abstract-Conference.html). NeurIPS 2025.
