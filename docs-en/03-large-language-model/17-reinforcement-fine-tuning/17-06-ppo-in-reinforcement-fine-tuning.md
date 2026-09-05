---
title: "17.6 PPO in Reinforcement Fine-Tuning"
chapter_title: "Reinforcement Fine-Tuning"
section_id: "17-06"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/17.强化微调/17.6 PPO算法在强化微调中的应用.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 17.6 PPO in Reinforcement Fine-Tuning

## I. PPO's Objective (the Same as in the Reinforcement Learning Part)

Policy objective:

$$
J^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)A_t,
\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t
\right)
\right]
$$

Here:

$$
r_t(\theta)=\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

- $r_t(\theta)$: the probability ratio between the new and old policies for action $a_t$.
- $A_t$: the estimated advantage of the current action, indicating how much better it is than average.
- $\epsilon$: the clipping range that limits policy-update magnitude.
Estimating $A_t$:

Looking only one step ahead, the $Q$ value of action $a_t$ can be estimated by $r_t + \gamma V(s_{t+1})$. The estimate of $A_t$ then equals the TD error:

$$
\hat{A}_t^{(1)}
=
\underbrace{r_t + \gamma V(s_{t+1})}_{\approx Q(s_t,a_t)}
- V(s_t)
=
\delta_t
$$

PPO uses GAE by default to calculate advantage, taking an exponentially weighted moving average of TD errors:

$$
\hat{A}_t^{\mathrm{GAE}}
=
\delta_t
+
(\gamma\lambda)\delta_{t+1}
+
(\gamma\lambda)^2\delta_{t+2}
+
\cdots
$$

Here, $\lambda$ is a hyperparameter in $[0,1]$.

The effect of TD errors is controlled through $\lambda$:

- At $\lambda = 0$: high bias, low variance.

$$
\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$A_t$ is exactly the current one-step TD error.

- Advantage: low variance, depending only on one random reward $r_t$.
- Disadvantage: high bias, strongly dependent on critic accuracy at $V(s_{t+1})$. A poorly trained critic produces incorrect estimates.
- At $\lambda = 1$: no bias, high variance.

$$
\hat{A}_t
=
\sum_{l=0}^{\infty}\gamma^l\delta_{t+l}
=
\left(\sum_{l=0}^{\infty}\gamma^l r_{t+l}\right)
- V(s_t)
$$

Expanding the expression cancels the intermediate $V$ terms, leaving Monte Carlo returns (actual returns) minus the baseline.

- Advantage: low bias; actual returns are the most accurate facts.
- Disadvantage: very high variance from accumulating randomness at every environment step.

## II. PPO Workflow in Reinforcement Fine-Tuning (Slightly Different from the Earlier Version)

1. Data collection

In traditional RL environments such as robot control, the agent interacts continuously for, say, 2,048 steps, producing a long trajectory that is later shuffled into minibatches for gradient calculation. PPO for LLM fine-tuning instead uses vectorized environments: multiple prompts can be processed in parallel, independently generating trajectories stored in a replay buffer.

“2,048 steps” is actually the total number of tokens generated across all parallel environments.

Suppose GPU parallelism allows $N$ prompts to be processed simultaneously (the number of parallel environments, such as $N = 16$).

- At $t = 1$, the model sees 16 different prompts and generates 16 tokens concurrently, one per prompt. This counts as 16 steps.
- At $t = 2$, it generates another 16 tokens based on the preceding tokens.
- The process continues.

The calculation is:

$$
\text{Total Steps }(2048)
=
\text{Number of parallel prompts}
\times
\text{Average response length}
$$

For example, “run 2,048 steps” in a diagram may mean:

1. The system samples 16 different prompts from the dataset.
2. The model generates answers to all 16 in parallel.
3. Each answer ends after an average of 128 tokens.
4. Total collected steps: $16 \times 128 = 2048$.

Thus, 2,048 steps contain complete stories for 16 prompts, not one extremely long story for a single prompt.

2. Advantage calculation (the same as traditional RL)

Initialization: initialize actor $\pi_\theta$ and critic $V_\phi$.

For each iteration:

1. Data collection (interaction):
   - Run current policy $\pi_{\theta_{\mathrm{old}}}$ for $T$ environment steps, such as 2,048.
   - Collect trajectories $\{s_t, a_t, r_t, s_{t+1}, \log \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)\}$.
   - Calculate state values $V(s_t)$ with the critic.
2. Advantage calculation:
   - Calculate TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
   - Calculate generalized advantage estimation (GAE) $A_t$, a recursive formula balancing bias and variance.
3. Optimization and policy synchronization

Collection: like a harvester, the GPU processes many prompts in parallel to fill a replay buffer of capacity 2,048 rapidly. It may contain question–answer records for 16 different problems.

Shuffle: thoroughly shuffle the 2,048 tokens (data points) to stabilize training and break correlations.

Minibatches: split the shuffled data into small batches, such as batch size 64.

- This gives $2048 / 64 = 32$ minibatches.
- The model uses those 32 minibatches to update its parameters.

3. Optimization:

- Important: $\pi_{\theta_{\mathrm{old}}}$ remains fixed as the denominator. The new $\pi_\theta$ in the numerator is optimized.
- Shuffle the $T$ collected data points into minibatches, such as 64 points each.
- Repeat for multiple epochs, such as 10:
  1. For each minibatch, calculate the new probability ratio $r_t(\theta)=\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\mathrm{old}}}(a \mid s)}$.
  2. Calculate the clipping loss $L^{\mathrm{CLIP}}$.
  3. Calculate critic value loss $L^{\mathrm{VF}} = (V_\phi(s_t) - V_{\mathrm{target}})^2$.
  4. Calculate entropy regularization $S$ to encourage exploration.
  5. Total loss: $L = -L^{\mathrm{CLIP}} + c_1 L^{\mathrm{VF}} - c_2 S$. Note the signs for gradient ascent/descent.
  6. Backpropagate to update $\theta$ and $\phi$.

4. Policy synchronization:

- After the round of updates, set $\pi_{\theta_{\mathrm{old}}} \leftarrow \pi_\theta$.
- Clear the data buffer and begin the next iteration.

## III. Why Do LLMs Choose PPO over SAC, and On-Policy over Off-Policy?

### (1) LLMs Output Discrete Rather than Continuous Actions

SAC's original design: SAC is a maximum-entropy RL algorithm designed for continuous action spaces. In robot control, actions are real-valued joint angles or torques. Discrete SAC variants exist, but high-dimensional discrete spaces are extremely expensive to handle.

The LLM setting: generation selects the next token from a huge vocabulary.

- Action-space size $|A|$: usually $32,000$ to $128,000$ or larger, such as the vocabularies of GPT-4 or Qwen.
- Curse of dimensionality: an actor–critic's critic evaluates $Q(s,a)$.
  - Discrete SAC generally marginalizes over all actions or calculates Softmax to obtain target $Q$ values and policy entropy.
  - Its formulas involve the expectation of $\pi(a' \mid s') \cdot [Q(s',a') - \alpha \log \pi(a' \mid s')]$, namely:

$$
\mathbb{E}_{a'\sim \pi(\cdot \mid s')}
\left[
Q(s',a')-\alpha\log\pi(a'\mid s')
\right]
=
\sum_{a'\in A}\pi(a'\mid s')
\left[
Q(s',a')-\alpha\log\pi(a'\mid s')
\right]
$$

Each update therefore requires a full probability calculation over $50,000+$ tokens, which is extremely costly.

One might suggest that a 100,000-token vocabulary already resembles a continuous space. Would changing the policy output from a 100,000-dimensional probability distribution to a continuous embedding vector not greatly reduce memory and compute? In practice, this approach has so far proved unworkable.

1. Blocked gradients: differentiating across a discrete cliff

This is the central mathematical obstacle.

- The illusion of continuous actions: suppose the actor outputs a continuous vector $\mathbf{v}\in\mathbb{R}^d$, such as $d=4096$. To provide it to the environment or predict the next token, it must be mapped back to the real vocabulary.
- Nearest-neighbor search: find the token embedding $\mathbf{e}_k$ closest to $\mathbf{v}$ in the embedding table:

$$
k
=
\underset{j\in V}{\arg\min}\,
\lVert \mathbf{v}-\mathbf{e}_j\rVert_2
$$

- Gradient difficulty: $\arg\min$ is nondifferentiable, or has zero gradient almost everywhere.
  - If $\mathbf{v}$ moves slightly without crossing a nearest-neighbor boundary, the nearest token remains $k$. Output and loss do not change, so the gradient is 0.
  - Crossing a boundary changes the nearest neighbor to token $m$, creating a step discontinuity with an infinite or undefined derivative.

Without gradients, backpropagation stops and the actor learns nothing. Relaxations such as Gumbel-Softmax exist, but have enormous variance and unstable training in a 100,000-dimensional space.

2. Empty regions: sparsity of the embedding space

One might ask, “Why use argmax at all? Why not feed continuous output $\mathbf{v}$ directly into the next network layer?”

This involves the manifold hypothesis.

- Word vectors are not uniformly distributed: in 4,096-dimensional space, valid token embeddings occupy only a tiny point cloud. Most of the space is invalid void space.
- Continuous-control difficulty: RL exploration usually adds Gaussian noise to actions. Adding noise to $\mathbf{v}$ will probably place it in an empty region.
  - Result: the vector resembles neither “apple” nor “banana,” but is disordered and semantically meaningless.
  - Language-model collapse: passing this meaningless vector onward makes Transformer attention process confused information, causing subsequent generation to become gibberish.

3. The efficiency paradox: KNN is slower than matrix multiplication

The continuous approach aims to improve efficiency, but may be slower.

- Discrete approach (current practice): the output layer is a linear transformation $W\cdot h_o$. Although $W$ is large, for example $4096\times100000$, it is just a large matrix multiplication, heavily optimized on GPUs through Tensor Cores.
- Continuous approach (proposed): converting the actor's output vector into a word for human readers or reward computation requires searching the entire vocabulary for the nearest neighbor.
  - Each generation step calculates distances to 100,000 embeddings.
  - This still amounts to multiplying $1\times4096$ by $4096\times100000$, plus sorting or $\arg\min$.
  - It is no faster than directly outputting logits and is slower because of additional indexing.

### (2) LLMs Are Better Suited to On-Policy than Off-Policy Learning

In traditional RL, such as robot control and game AI, much effort goes into off-policy learning because sample efficiency is considered essential.

For LLMs, on-policy learning is not merely a compromise; it can be a core advantage, overturning traditional RL rules of thumb. This can be analyzed mathematically, computationally, and through alignment objectives.

1. Mathematical breakdown: the curse of importance sampling

Off-policy learning uses data from an old policy to update a new one. Correcting distribution differences generally requires importance-sampling (IS) weights:

$$
\rho_t
=
\frac{\pi_{\mathrm{new}}(a_t \mid s_t)}
{\pi_{\mathrm{old}}(a_t \mid s_t)}
$$

In robot control, action sequences are usually short or action spaces simple, keeping $\rho_t$ manageable. In LLMs, the situation becomes uncontrolled:

- Multiplication across sequence length: a trajectory usually contains hundreds or thousands of tokens. Its probability ratio is the product of per-step ratios:

$$
\rho_{\mathrm{trajectory}}
=
\prod_{t=1}^{T}
\frac{\pi_{\mathrm{new}}(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}
$$

Off-policy algorithms update new policies using old-policy data. Depending on value estimation, there are two cases:

(1) Without a Q network

The new-policy objective is:

$$
J(\pi_{\mathrm{new}})
=
\mathbb{E}_{\tau\sim\pi_{\mathrm{new}}}[R(\tau)]
=
\int P(\tau\mid\pi_{\mathrm{new}})R(\tau)d\tau
$$

Since only $\pi_{\mathrm{old}}$ data is available, importance sampling is required:

$$
J(\pi_{\mathrm{new}})
=
\mathbb{E}_{\tau\sim\pi_{\mathrm{old}}}
\left[
\frac{P(\tau\mid\pi_{\mathrm{new}})}
{P(\tau\mid\pi_{\mathrm{old}})}
R(\tau)
\right]
$$

Expanding trajectory probability $P(\tau)$ cancels the environment dynamics, leaving action-probability ratios:

$$
\frac{P(\tau\mid\pi_{\mathrm{new}})}
{P(\tau\mid\pi_{\mathrm{old}})}
=
\frac{\prod_{t=0}^{T}\pi_{\mathrm{new}}(a_t\mid s_t)}
{\prod_{t=0}^{T}\pi_{\mathrm{old}}(a_t\mid s_t)}
=
\prod_{t=0}^{T}
\frac{\pi_{\mathrm{new}}(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}
=
\prod_{t=0}^{T}\rho_t
$$

This is where the product arises:

$$
\text{Off-Policy Gradient}
\propto
\mathbb{E}
\left[
\left(\prod_{t=0}^{T}\rho_t\right)R(\tau)
\right]
$$

For LLMs, $T\approx1024$. Even small deviations of $\rho_t$ from 1 make the product fluctuate sharply between 0 and $+\infty$, exploding variance. With $T=1024$, even small per-step differences give:

- Ratio $1.01$: $1.01^{1024}\approx 26,739$.
- Ratio $0.99$: $0.99^{1024}\approx 0.00003$.

IS-weight variance becomes nearly uncontrollable. PPO's on-policy advantage is that it enforces $\pi_{\mathrm{new}}\approx\pi_{\mathrm{old}}$, limits policy drift through clipping and short rollouts, and keeps $\rho_t\approx1$, avoiding long-sequence numerical explosions.

(2) With a Q network: avoiding exploding variance but risking exploding bias

Generating an LLM response usually takes $T=1000$ steps. Suppose the Q network has a small estimation error $\epsilon$, nearly inevitable with a 70B-parameter model and sparse rewards.

$$
Q_{\mathrm{target}}(s_{T-1})
=
r_{T-1}
+\mathrm{EndValue}
+\epsilon
$$

$$
Q_{\mathrm{target}}(s_{T-2})
=
r_{T-2}
+\gamma Q_{\mathrm{target}}(s_{T-1})
+\epsilon
$$

Recursing backward accumulates error along the long sequence:

$$
Q_{\mathrm{target}}(s_0)
=
\sum_{t=0}^{T-1}\gamma^t r_t
+
\sum_{k=0}^{T-1}\gamma^k \epsilon
$$

Because of Q-value overestimation, $\epsilon$ is often positive, and its accumulation undermines critic judgments.

2. Reversed computational architecture: recomputation costs more than generation

In traditional RL, environmental interaction takes physical time while neural computation is fast, so stored data should be reused repeatedly on GPUs.

In LLM RLHF, interaction (text generation) is itself neural computation and is extremely expensive.

Using old data with off-policy methods such as SAC or IMPALA requires knowing its probability distribution under current parameters to compute $\log\pi_{\mathrm{new}}$ or KL divergence. This requires another forward pass of the old prompt and response through the current 70B model.

3. Distinctive alignment objectives: fine-tuning rather than seeking a global optimum

This is the most fundamental difference.

- Traditional RL, such as AlphaGo: seek a global optimum, discovering brilliant moves humans have never seen. Off-policy learning retains accidentally discovered unusual high-reward actions in a replay buffer and repeatedly reinforces them.
- LLM alignment (RLHF): constrained fine-tuning. The goal is not to invent a new language or hack rewards through gibberish, but to remain near the SFT model with small style and safety improvements.
  - Off-policy risk: exploring regions far from the current policy can cause model collapse, in which the model speaks nonsense to maximize reward.

Off-policy learning's tendency to deviate comes from its own characteristics.

Mechanism: store all historical transitions $(s,a,r,s')$ in a replay buffer and sample them randomly during training.

Consequence: the buffer may contain data from $\pi_{\mathrm{old}}$ 100 epochs earlier.

- That policy may have been highly random, accidentally trying an extreme action early in exploration. In an LLM, this could be an obscure gibberish word mistakenly given a high score by the reward model.
- Although mature $\pi_{\mathrm{new}}$ would never intentionally produce it, off-policy sampling retrieves it and notices, “This action scored 100!”
- The update then pulls the current policy toward that historical, unreasonable high-reward region.

By contrast, on-policy PPO discards data after use. If an unusual high-scoring error occurs once and does not recur, PPO quickly forgets it rather than repeatedly revisiting it.

On-policy learning calculates the policy gradient under the current policy and takes one step to adjust the distribution, generally staying near the existing solution. The “Phenomenon Explanation and Theoretical Analysis in Large-Model Research” part also proves that on-policy methods repeatedly project between feasible and optimal policies. The limiting policy is theoretically the optimal policy with the smallest KL divergence from the initial policy.

Overestimating unseen regions is another reason for deviation.

Q-value generalization error: neural networks generalize, so a Q network predicts values even for unseen state–action pairs $(s,a_{\mathrm{unseen}})$ that current policy $\pi$ has never visited.

Overestimation tendency: without real-data constraints on these $(s,a_{\mathrm{unseen}})$ pairs, the Q network's predictions often have high variance. The $\max$ in $y=r+\gamma\max Q$ specifically selects incorrectly overestimated values.

Result: the Q network effectively imagines, “I have never been there, but it must be full of gold.”

Action: the actor is drawn toward regions far from the current distribution that may not actually yield high returns.

### (3) LLM Fine-Tuning Requires Controlled Update Sizes

LLM fine-tuning must control update magnitudes. Excessively large parameter updates rapidly cause “language collapse”: maximizing reward destroys pretrained language ability, yielding gibberish or repeated meaningless characters, another form of reward hacking.

PPO limits the difference between new and old policies through clipping:

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right]
$$

Here, $r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$ is the probability ratio.

- Role: $\epsilon$, usually 0.2, prevents $\pi_\theta$ from moving too far from $\pi_{\theta_{\mathrm{old}}}$.
- Result: while learning human preferences, the LLM retains pretrained abilities such as grammar and logic.

Besides limiting one update through clipping, modern large models often add a KL term relative to a reference distribution (the pretrained model's output distribution), preventing many small updates from accumulating excessive drift.

SAC maximizes entropy alongside cumulative reward:

$$
J(\pi)
=
\sum_{t=0}^{T}
\mathbb{E}_{(s_t,a_t)\sim\rho_\pi}
\left[
r(s_t,a_t)
+\alpha\mathcal{H}(\pi(\cdot\mid s_t))
\right]
$$

Risk: although entropy encourages exploration, RLHF usually seeks confident, logically rigorous answers rather than high-entropy random tokens for exploration. Excessive entropy regularization can make LLM output divergent and incoherent.

### (4) Q-Network Stability Under Sparse Rewards

- SAC (tightly coupled actor and critic): SAC strongly depends on Q-network critic accuracy. Training an LLM critic to judge whether a sentence is good, effectively a proxy for the reward model, is already difficult. A divergent or inaccurate critic quickly corrupts the actor.
  - Under large-scale sparse rewards, bootstrapping in Q-learning methods easily produces overestimation bias.
- PPO (Monte Carlo-like regression): PPO uses generalized advantage estimation (GAE) to calculate $\hat{A}_t$. Although it also uses a critic (value network), it relies more on actual returns over the whole trajectory. Compared with SAC's strongly bootstrapped TD learning, PPO is usually more robust for long-sequence generation with long episodes.

## References

- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). arXiv:1506.02438.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). ICML 2018.
- Haarnoja, T., Zhou, A., Hartikainen, K., et al. (2018). [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905). arXiv:1812.05905.
