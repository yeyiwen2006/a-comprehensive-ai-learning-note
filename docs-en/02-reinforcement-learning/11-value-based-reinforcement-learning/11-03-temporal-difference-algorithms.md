---
title: "11.3 Temporal-Difference Algorithms"
chapter_title: "Value-Based Reinforcement Learning"
section_id: "11-03"
language: en
source_language: zh
source_docx: "第2部分 强化学习/11.基于价值的强化学习/11.3 时序差分算法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 11.3 Temporal-Difference Algorithms

Temporal difference estimates a policy's value function by combining ideas from Monte Carlo and dynamic programming. Like Monte Carlo, it learns from sampled data without prior knowledge of the environment, specifically through “sampling plus a moving average.” Like dynamic programming, it follows the Bellman equation's idea of using the “next state's” estimated value to update the “current state's” estimated value.

## I. Sarsa

1. Original Sarsa

Since temporal difference can estimate a value function, a natural question is whether reinforcement learning can proceed similarly to policy iteration. Temporal difference already provides policy evaluation, but how can policy improvement occur without known reward and transition functions? The answer is to estimate action values $Q$ directly with temporal difference:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]
$$

Then greedily select the highest-value action in each state, $\arg\max_a Q(s,a)$. This appears to form a complete RL algorithm: select actions greedily from action values to interact with the environment, then update the estimates with temporal difference using the resulting data.

However, two issues require further consideration. First, accurate temporal-difference estimation of a policy's state values requires enormous numbers of samples. In practice, we can ignore this and evaluate with only some samples before updating the policy, because improvement can occur before evaluation is complete. Recall that value iteration (Section 4.4) does exactly this, reflecting generalized policy iteration. Second, always producing a deterministic greedy policy may prevent some state-action pairs $(s,a)$ from ever appearing in sampled sequences. Their values then cannot be estimated, so policy improvement is not guaranteed to improve the policy. Chapter 2 discussed this in detail. A simple, common solution replaces pure greediness with an $\epsilon$-greedy policy: choose the highest-value action with probability $1-\epsilon$ and a random action with probability $\epsilon$:

$$
\pi(a|s)=
\begin{cases}
\epsilon/|A|+1-\epsilon, & \text{if }a=\arg\max_{a'}Q(s,a')\\
\epsilon/|A|, & \text{other actions}
\end{cases}
$$

2. n-step Sarsa

Monte Carlo uses every subsequent reward without value estimates, whereas temporal difference uses only one reward and the next state's estimated value. What distinguishes them? Generally, Monte Carlo is unbiased but has relatively high variance: every state transition is uncertain, and the different rewards from actions at each step accumulate, greatly affecting the final value estimate. Temporal difference has very low variance because it considers only one transition and reward, but is biased because it uses the next state's estimated rather than true value. Can their strengths be combined? Yes: multistep temporal difference! Use $n$ rewards and then the subsequent state's value estimate. Replace

$$
G_t=r_t+\gamma Q(s_{t+1},a_{t+1})
$$

with

$$
G_t=r_t+\gamma r_{t+1}+\cdots+\gamma^n Q(s_{t+n},a_{t+n})
$$

The corresponding multistep Sarsa algorithm replaces the action-value update in Sarsa (see Section 5.3),

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]
$$

with

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[r_t+\gamma r_{t+1}+\cdots+\gamma^n Q(s_{t+n},a_{t+n})-Q(s_t,a_t)]
$$

## II. Q-Learning

Another famous temporal-difference RL algorithm besides Sarsa is Q-learning. Its main difference from Sarsa is the update:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[R_t+\gamma\max_a Q(s_{t+1},a)-Q(s_t,a_t)]
$$

## III. Online and Offline Learning Policies

1. Core difference

The policy pi_b used to sample data is called the behavior policy, while the policy pi_t updated using those data is the target policy.

(1) Online learning (Sarsa as an example): the target and behavior policies coincide; the policy being optimized is its own behavior policy. Learning (data sampling) is therefore tightly bound to that policy. Once the policy changes, old-policy data become invalid and must be refreshed in real time (for example, A' in Sarsa's tuple (S, A, R, S', A') is “signed” by pi_{old}). In classic cliff walking, an earlier foolish policy might often select “jump off the cliff” as A' at cliff-edge state S'. A smarter current policy almost always selects “take the safe path.” Updating its current Q values using old data containing the “jump” A' incorrectly lowers the score. More informally, if the agent is a drunkard (current policy pi) seeking to evaluate its own journey, watching a tightrope master's video might suggest “this route is safe.” Believing that could lead it off the cliff. To evaluate its own fate accurately, it must use data it generates itself in real time.

(2) Offline learning (Q-Learning as an example): target and behavior policies differ. Training behavior often explores the environment thoroughly, whereas updates optimize only the best actions and ignore other behavior. With sufficient exploration, the optimal policy depends only on the environment's Markov properties, not on the agent's behavior policy; only objective environmental regularities need to be learned. Offline data (such as Q-Learning's tuple (S,A,R,S’)) therefore concern only the environment. If I seek objective truth, anyone's data are useful. In board games, for example, millions of winning and losing game records can be downloaded and fed to Q-Learning. Seeing a novice fall into a trap reveals where not to go; seeing an expert succeed reveals a shortcut.

Note: policy-based reinforcement learning is generally always online, but the converse need not hold.

2. Comparing Sarsa and Q-Learning

**Q-Learning: removing the “person” to reveal the “world”**

Consider the tuple $(S,A,R,S')$ again.

- $S\to A\to R,S'$.
- What happens in this step depends entirely on the environment's physical engine (such as Newton's second law or game logic).
- It contains no information about “what you intend to do after $S'$.”

**Q-Learning's great strength:** it constructs an update target using only this purely physical tuple and a purely mathematical assumption (max).

$$
Target=\underbrace{R}_{\text{physical outcome}}+\underbrace{\gamma\max_a Q(S',a)}_{\text{mathematical assumption}}
$$

No “person (policy)” appears in this formula. That is why it describes “objective physical regularities of the environment”: it maps the environment's “value landscape.” The map exists whether or not you play, and the peak's height (the optimum) is unchanged.

**Sarsa: person and world are inseparable**

Consider the tuple $(S,A,R,S',A')$ again.

- $A'$ is a choice made by the person.
- Sarsa's target is:

$$
Target=\underbrace{R}_{\text{physical outcome}}+\underbrace{\gamma Q(S',A')}_{\text{human behavior}}
$$

Here the “person ($A'$)” enters value evaluation. An analogy:

- Physical property (Q-Learning): a Ferrari's maximum speed is 350 km/h, a property of the car independent of how you drive.
- Sarsa expectation: your average speed in the Ferrari is 60 km/h because you dare not drive fast or traffic is congested. This combines properties of the car and driver.

- Sarsa fits the expected outcome of the joint effects of “physical environment $P$” and “behavior policy $\pi$.”
  - Formula: $E_{s'\sim P,a'\sim\pi}[R+\gamma Q(s',a')]$
- Q-Learning fits the theoretical limit allowed by “physical environment $P$.”
  - Formula: $E_{s'\sim P}[R+\gamma\max_{a'}Q(s',a')]$

3. Illustrating their tendencies with cliff walking

**Q-Learning says: I do not care what foolish action you actually take next; I consider only the theoretical maximum.**

- Situation: you stand at the cliff edge $S'$.
- Actual action: because your hand slips (exploration), you choose $a_2$ and jump.
- Q-Learning's calculation: when updating values from the previous step or two, it ignores that you jumped.
  - It examines all actions at $S'$: $a_1$ (walk back, 100 points), $a_2$ (jump, -100 points).
  - It selects the maximum: $\max(100,-100)=100$.
  - Conclusion: it regards state $S'$ as worth 100.

Its underlying message is: “You slipped and jumped, but that was an execution error. Theoretically, a sufficiently rational person can walk back from the edge, so standing at the edge remains a good state.”

This is an “offline policy”: it estimates values using the optimal policy (Max), rather than judging actions through the current behavior policy (including random exploration).

**Sarsa says: your misfortune next must be charged to the present.**

- Situation: you stand at the cliff edge $S'$.
- Actual action: your hand slips (exploration), and you choose $a_2$ and jump.
- Sarsa's calculation: it carefully observes what you actually chose.
  - It sees $a_2$ (jump, -100 points).
  - It directly uses the value of this $a_2$: -100.
  - Conclusion: it regards state $S'$ as worth -100.

Its underlying message is: “Do not talk to me about the theoretical optimum. In reality, your hand easily slips at the cliff edge. Since your shaky hand (exploration probability) makes this state frequently fatal, standing there is itself bad and deserves a low score.”

This is an “online policy”: it must evaluate values using behavior that actually occurs.

**Q-Learning (off-policy)**

- Learns an optimal path hugging the cliff edge.
- Reason: it assumes perfect behavior (Max), without slips. Its values are based on the optimal solution that “never falls.”
- Actual behavior: during training, $\epsilon$-greedy exploration still randomly sends it over the cliff with 10% probability. It learns the optimal route but falls frequently while training.

**Sarsa (on-policy)**

- Learns a safe path farther from the cliff.
- Reason: it knows it may jump randomly with probability $\epsilon$. A random lapse at the edge is fatal, so Sarsa assigns that state low value (high risk).
- Actual behavior: although the route is not theoretically shortest, execution is safer (especially with randomness) and cumulative reward may be higher.

4. If Sarsa depends on a policy, why is it not policy-based?

Whether an algorithm is “value-based” or “policy-based” depends not on whether its “data source” depends on a policy, but on what it “ultimately learns.”

Although Sarsa depends on the policy during learning (on-policy), its central maintained and updated object is a $Q$ table (or $Q$ network), not policy parameters.

**Policy-based**

- Train a neural network $\pi_\theta(a|s)$ with parameters $\theta$.
- Given a state, it directly outputs action probabilities (for example, left 0.8, right 0.2).
- Adjust $\theta$ to increase good actions' probabilities.
- There is no $Q$ table (or $Q$ is not central).

**Value-based: Sarsa belongs here**

- Fill a $Q$ table (or train a $Q$ network), with $Q$ values as parameters.
- Given a state and action, the network outputs “how many points this action is worth.”
- Where is the policy? It is implicit, a “byproduct” of the $Q$ table.
- Sarsa's logic: I am responsible only for accurate $Q$ values. The policy is a simple “table lookup”: choose the highest $Q$ (or use $\epsilon$-greedy).
- Conclusion: Sarsa's central task is “estimating value,” not “adjusting policy parameters,” so it is value-based.

In Sarsa, policy and value have a “master–servant” relationship:

- Master: value function $Q$, the central asset updated iteratively with much effort.
- Servant: policy $\pi$, which has no independent thinking and merely carries $Q$ values into action.
  - Sarsa's policy is usually $\pi(s)=\arg\max_a Q(s,a)$ (with some random noise).
  - Once $Q$ changes, the policy changes automatically; no dedicated policy gradient is needed.

In policy-based methods (such as REINFORCE):

- Master: policy $\pi_\theta$. Its logic is optimized directly.
- A value function may not exist or may merely help reduce variance (as in actor–critic).

**Q-Learning (off-policy)**

- Aims to estimate $Q^*$ (the optimal policy's value).
- Motto: “I do not care how you play now; I calculate the theoretical best possible score.”

**Sarsa (on-policy)**

- Aims to estimate $Q^\pi$ (the current policy's value).
- Motto: “I calculate the score you will actually get if you continue playing as timidly as you do now.”

The key: whether calculating the “theoretical best score” or “current actual score,” both calculate “scores,” or values. If calculating scores is the central task, the method is value-based.

5. Improving Sarsa's data utilization

**Method A: multistep updates (N-step Sarsa)**

Sarsa need not update after every step. It can take $n$ steps before updating.

- Accumulate: $S_1\to S_2\to S_3\to\cdots\to S_n$.
- Then update the values of all points along the path in one backward sweep.
- This uses the entire data sequence from an interaction, rather than updating only its first point.

**Method B: eligibility traces, Sarsa($\lambda$)**

Eligibility traces are Sarsa's fully developed form.

- Principle: although data from a much older policy cannot be used, there is short-term memory of the path within one task.
- Mechanism: mark every visited $(S,A)$ pair with an “activity-level” trace.
- Result: upon reaching the end and receiving a reward, update not only the last step but all relevant recent steps by following the traces backward.
- To address your question: in Sarsa($\lambda$), data are not discarded; traces preserve their short-term influence, allowing values at other positions and actions to be updated together.

## References

- [Hands-on Reinforcement Learning (translated title; in Chinese)](https://hrl.boyuai.com/).
- Watkins, C. J. C. H., & Dayan, P. (1992). [Q-learning](https://doi.org/10.1007/BF00992698). Machine Learning, 8, 279–292.
- Rummery, G. A., & Niranjan, M. (1994). [On-Line Q-Learning Using Connectionist Systems](https://www.cs.utexas.edu/~shivaram/readings/b2hd-RummeryNiranjan1994.html). Cambridge University Engineering Department, Technical Report CUED/F-INFENG/TR 166.
