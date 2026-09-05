---
title: "14.4 Multi-Agent Reinforcement Learning"
chapter_title: "Other Reinforcement Learning Algorithms"
section_id: "14-04"
language: "en"
source_language: "zh"
source_docx: "第2部分 强化学习/14.强化学习的其他算法/14.4 多智能体强化学习.docx"
status: "translated"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 14.4 Multi-Agent Reinforcement Learning

## I. Objective and Basic Methods

### (I) Objective: Learn a Policy $\pi_i$ for Each Agent to Maximize Its Own Expected Cumulative Reward.

### (II) Basic Methods

| Paradigm | Fully centralized | Fully decentralized |
| --- | --- | --- |
| Principle | Treat all agents as a single “super-agent.” The input is the joint state, and the output is the joint action. | Assume that agents are independent, with each treating the others as part of the environment. |
| Advantages | The environment becomes stationary, with good theoretical convergence. | Highly scalable, unrestricted by the number of agents, and avoids the curse of dimensionality. |
| Disadvantages | Dimensional explosion: state/action spaces grow exponentially with the number of agents, making training difficult. | Nonstationarity: convergence is not guaranteed because other agents' policy changes are ignored. |

## II. Independent PPO (IPPO)

IPPO is fully decentralized. It directly trains each agent using that agent's own Proximal Policy Optimization (PPO) algorithm. Although convergence cannot be guaranteed theoretically, it often works well in practice and scales well.

IPPO workflow:

1. Initialization: separately initialize policy network $\pi_i$ and value network $V_i$ for each of $N$ agents.
2. Training loop (for each training round):
   * Data collection: all agents run simultaneously in the environment, each collecting a trajectory.
   * Advantage estimation: for each agent $i$, compute advantage $\hat{A}_i$ using its value network $V_i$ and generalized advantage estimation (GAE).
   * Policy update: for each agent $i$, maximize the PPO-Clip objective to update policy $\pi_i$.
   * Value update: for each agent $i$, minimize mean squared error to update value network $V_i$.
3. End the loop.

## III. MADDPG (OpenAI, 2017)

### (I) The CTDE Paradigm

To balance environmental nonstationarity and scalability, researchers proposed the centralized training with decentralized execution (CTDE) paradigm.

1. Training: global information is available (all agents' states, actions, and rewards). This resembles a football coach with a bird's-eye view guiding players' cooperation.

2. Execution: agents decide using only local observations. Like players on the field, they act from what they see locally and no longer depend on real-time instructions from the coach.

### (II) Model Architecture

Multi-Agent DDPG (MADDPG) is a representative CTDE algorithm based on an Actor-Critic architecture.

* Actor (policy network): takes only local observation $o_i$ as input and outputs action $a_i$. Used during execution.
* Critic (value network): takes global information $x$ (containing all observations $o_1,\ldots,o_N$) and all actions $a_1,\ldots,a_N$. Used only during training.

The mathematical reasoning and gradient formulas follow.

Define the global state information as $x$ and all agents' actions as $a=(a_1,\ldots,a_N)$.

1. Critic update (centralized part): each agent $i$'s Critic network $Q_i^{\mu}(x,a_1,\ldots,a_N)$ evaluates the value of taking a joint action in the global state. Its loss is mean squared error:

$$
L(\theta_i)
=
\mathbb{E}_{x,a,r,x'}
\bigl[
\bigl(
Q_i^{\mu}(x,a_1,\ldots,a_N)-y
\bigr)^{2}
\bigr]
$$

Target $y$ is calculated using target networks:

$$
y
=
r_i
+
\gamma
Q_i^{\mu'}\bigl(
x',
a_1',
\ldots,
a_N'
\bigr)
\quad
\text{where }
a_j'=\mu_j'(o_j')
$$

Note: $Q_i^{\mu'}$ is the target Critic and $\mu_j'$ the target Actor.

2. Actor update (the basis for decentralized execution): for deterministic policy $\mu_{\theta_i}$, maximize the $Q$ value to update the policy. Because $Q$ is centralized, it can tell the Actor whether an action is good given its teammates' actions. The deterministic policy gradient is:

$$
\nabla_{\theta_i}J(\mu_i)
=
\mathbb{E}_{x,a\sim\mathcal{D}}
\bigl[
\nabla_{\theta_i}\mu_i(o_i)
\cdot
\nabla_{a_i}Q_i^{\mu}(x,a_1,\ldots,a_N)
\big|_{a_i=\mu_i(o_i)}
\bigr]
$$

Explanation:

* $\nabla_{\theta_i}\mu_i(o_i)$: how the Actor can change its parameters to change its output action.
* $\nabla_{a_i}Q_i^{\mu}(\cdots)$: the Critic tells the Actor which direction of action change increases the $Q$ value, namely the global payoff.

### (III) Training Workflow

1. Initialization:
   * Initialize Actor $\mu_i$ and Critic $Q_i$ for each agent $i$.
   * Initialize corresponding target networks $\mu_i'$ and $Q_i'$.
   * Initialize replay buffer $\mathcal{D}$.
2. Main loop (for sequences $1$ to $M$):
   * Initialize the environment state (for data collection) and random noise process $\mathcal{N}$.
   * Step loop (for $t=1$ to $\mathrm{max\_steps}$):
     1. Decision: each agent $i$ selects an action from local observation $o_i$: $a_i=\mu_i(o_i)+\mathcal{N}_t$.
     2. Interaction: execute joint action $a=(a_1,\ldots,a_N)$; the environment returns rewards and new observations $x'$.
     3. Storage: store transition tuple $(x,a,r,x')$ in replay buffer $\mathcal{D}$.
     4. Training: once the buffer is large enough, randomly sample a batch from $\mathcal{D}$.
     5. Critic update: calculate target $y$ and minimize the loss to update each agent's $Q_i$. Training is centralized because the Critic takes all agents' actions as input.
     6. Actor update: update each agent's $\mu_i$ by gradient ascent. The Actor outputs actions from local observations, while the centralized Critic guides the gradient direction.
     7. Soft update: update target networks using parameter $\tau$, then use the new observations as the next state and continue sampling.
3. End the main loop.

The target-network soft update is:

$$
\theta'
\leftarrow
\tau\theta+(1-\tau)\theta'
$$

As in other Actor-Critic algorithms, after training, discard the Critic and retain the Actor for forward inference using its trained, “frozen” policy. Inference after training is therefore fully decentralized.

## IV. Multiple Agents with an Orchestrator

Take Kimi K2.5 Agent Swarm as an example. During inference, when a task is supplied, an orchestrator determines the number and roles of the subagents it requires (providing roles to agents as context), assigns tasks, and schedules the agents. Each subagent is an “instance” of the same large model with identical weights.

Training: first train the subagent (the large model). After subagent training is complete, freeze its parameters and train the orchestrator through end-to-end reinforcement learning (initially using small models as subagents, then replacing them with large models).

## References

- [Hands-on Reinforcement Learning (translated title; in Chinese)](https://hrl.boyuai.com/).
- Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275). NeurIPS 2017.
- de Witt, C. S., Gupta, T., Makoviichuk, D., et al. (2020). [Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?](https://arxiv.org/abs/2011.09533). arXiv:2011.09533.
- Kimi Team. (2026). [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276). arXiv:2602.02276.
