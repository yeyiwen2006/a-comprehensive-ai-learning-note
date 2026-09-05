---
title: "13.1 Actor-Critic"
chapter_title: "Combining Value and Policy Methods"
section_id: "13-01"
language: en
source_language: zh
source_docx: "第2部分 强化学习/13.综合价值与策略的算法/13.1 Actor-Critic算法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 13.1 Actor-Critic

## I. Basic Principles of Actor-Critic

1. Background

Policy gradient algorithms obtain the adjustment direction for parameters θ from the policy gradient, then determine its sign and magnitude through a Monte Carlo action-value estimate G_t. Monte Carlo requires many samples, complete trajectories before updates, and has high variance. Since G_t itself estimates how good an action value is, we can combine the earlier value-based methods by introducing an action-value function.

2. Mathematical representation

To maximize the objective (expected Q value under the state-action distribution), the policy gradient should be:

Recall the earlier conclusion: the ideal policy gradient is:

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a\mid s)\cdot Q^\pi(s,a)\right]
$$

To reduce variance, introduce baseline $V^\pi(s)$ and convert $Q$ into an advantage function:

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

Here Q(s,a) is the value of taking a at s, V(s) the state's value, and A(s,a) action a's advantage over the average.

By the Bellman equation, if a sample next reaches s’, the next update assumes:

According to the Bellman equation:

$$
Q(s,a)\approx r+\gamma V(s')
$$

Substitute this into the advantage formula:

$$
A(s,a)\approx \underbrace{r+\gamma V_\phi(s')-V_\phi(s)}_{\text{TD Error }\delta}
$$

Thus no Q network is needed; a V network suffices.

3. Network updates

Unlike Sarsa: no “Q table”

Actor-critic usually consists of a policy network and a value network, separately optimizing policy and value functions to combine their advantages.

(1) Actor policy-network update

The policy network updates using the gradient above, replacing G_t with A(s,a) under the policy. Because G_t is unnecessary and A(s,a) stored by the value network can be used directly, updates can occur at every step. Positive A(s,a) tends to increase a's probability; negative values tend to decrease it.

(2) Critic value-network update

After transitioning from s to s’ and receiving r, the update target for V(s) is Target=r+gamma*V(s’).

4. Comparison with other algorithms (Sarsa and DQN)

(1) Comparison with Sarsa

Sarsa's Q both scores and decides actions (epsilon-greedy), whereas the critic's Q only scores, leaving policy control to the actor. Sarsa has an explicit Q table; the critic fits values with deep learning.

Shared foundations:

* Same mathematical basis: both use the Bellman expectation equation, $\mathrm{Value}\approx r+\gamma\cdot\mathrm{Next\_Value}$. Here $\mathrm{Next\_Value}$ means “value from continuing under the current policy,” not a maximum.
* Same policy property: both are usually on-policy. Their targets closely track actual agent behavior; if the agent becomes less capable, their value estimates fall too.

| Dimension | Critic (usually V-based) | Sarsa (Q-based) |
| --- | --- | --- |
| Required data | $S,A,R,S'$ | $S,A,R,S',A'$ |
| Update timing | Can update once $S'$ is observed; only the average prospects of $S'$ need to be estimated. | Must wait until $A'$ is chosen; it needs the exact next route. |
| Update target | $Target=r+\gamma V(s')$, directly using state value and implicitly averaging over actions. | $Target=r+\gamma Q(s',a')$, using the value of the actual next action. |
| Randomness/variance | Lower. $V(s')$ is a smooth network output filtering out randomness in next-action selection. | Higher. The specific next action $a'$ matters; a good step followed by a poor one makes the target fluctuate. |
| Core role | Supporting coach. Only scores and must work with an actor; produces no actions itself. | Sole decision-maker and doer. Both scores and selects actions directly from scores ($\epsilon$-greedy). |

(2) Comparison with DQN

The critic and Sarsa use Bellman expectation equations, whereas DQN uses Bellman optimality. The critic and Sarsa estimate expected values under the current policy's distribution, requiring fresh current-policy data. Specifically:

Sarsa:

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left[\underbrace{r+\gamma Q(s',a')}_{\text{Target}}-Q(s,a)\right]
$$

Critic:

$$
L=\left(\underbrace{r+\gamma V(s')}_{\text{Target}}-V(s)\right)^2
$$

Although policy is not explicit, this update makes V(s) essentially a moving average of Q(s,a), namely r+gamma*V(s’), from past actions. As policy changes, action probabilities change. At the next step, a new action a must therefore be sampled under the current policy to ensure the expectation of r+V(s’) is an unbiased estimate of that new policy's value. Actions sampled under other policies cannot update V, making this on-policy.

DQN's update, however, is policy-independent and uses only optimal value.

| Property | Critic (in AC) | Sarsa | DQN |
| --- | --- | --- | --- |
| Core equation | Bellman expectation | Bellman expectation | Bellman optimality |
| Target | $r+\gamma V(s')$ | $r+\gamma Q(s',a')$ | $r+\gamma\max Q(s',a')$ |
| View of the future | Honest (current probabilities) | Honest (actual choices) | Optimistic (optimal future assumed) |
| Required sample | $(s,a,r,s')$ | $(s,a,r,s',a')$ | $(s,a,r,s')$ |
| Data use | On-policy (usually not reused) | On-policy (usually not reused) | Off-policy (replay buffer reusable) |
| Main purpose | Reduce PG variance (PPO/A2C) | Learn safe, conservative policies | Learn policies seeking the optimum |

Scenario: a path hugs a cliff. Falling costs 100 points, while a normal step costs 1 point.

Policy characteristic: current $\pi$ is still somewhat clumsy, with a $10\%$ chance of slipping off ($\epsilon$-greedy).

DQN (Q-Learning) perspective:

* “The cliff-edge route is shortest! Provided I do not slip (take max), it is optimal.”
* Result: its learned value function encourages walking along the edge. But because you actually do slip, you keep falling.
* Characteristic: overconfident, evaluating the “theoretically perfect policy.”

SARSA (and the critic in AC) perspective:

* “The cliff-edge route is shortest, but with your current clumsiness (policy $\pi$), you are likely to fall and die there. Its value ($Q$ or $V$) is actually low!”
* Result: its value estimate pushes the actor away from the cliff onto a safer path.
* Characteristic: realistic, evaluating “your current actual ability.”

5. Components during training and forward inference

All architectures with an actor and critic, including TRPO, PPO, DDPG, and SAC introduced later, follow these principles:

Training: actor and critic work together.

Forward inference: discard the critic and retain only the actor.

When learning to drive (training), a coach (critic) sits beside you, continually scoring and correcting your actions (updating the policy). Once licensed and driving alone (inference), the coach is absent and only you (actor) drive. You simply observe road conditions (state) and steer (action); actions no longer need correction because the policy no longer updates.

## II. A2C and A3C

As discussed, this is on-policy and cannot use a replay buffer like SAC (see the SAC section). Every policy update requires resampling. This is ill-suited to parallel computation and creates serious data correlation: nearby times have similar states s, and visitation changes over training, violating the i.i.d. assumption. Different workers must therefore sample simultaneously and combine gradient updates. Two underlying implementations follow.

1. A3C (Google DeepMind, ICML 2016)

Each worker is an independent process containing an environment copy and a local network copy. It runs local forward propagation, computes gradients, pushes them to a global network, and retrieves the latest global parameters to overwrite its local network.

2. A2C (OpenAI, 2017)

Uses multiprocessing, known as SubprocVecEnv (subprocess-vectorized environments). Each worker process holds only an environment copy; the main process holds the sole neural network (usually on a GPU). Sixteen workers execute env.step() in parallel and return their states. The main process collects the 16 states s into a batch, computes their 16 actions in one GPU pass, and sends them back. Workers are lightweight and only simulate the physical environment.

Current mainstream actor-critic algorithms, including PPO, mostly retain A2C's “vectorized environment” parallel architecture.

## References

- Mnih, V., Badia, A. P., Mirza, M., et al. (2016). [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783). ICML 2016.
- OpenAI. (2017). [OpenAI Baselines: ACKTR & A2C](https://openai.com/index/openai-baselines-acktr-a2c/).
