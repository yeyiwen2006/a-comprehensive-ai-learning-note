---
title: "14.2 Model-Based Reinforcement Learning"
chapter_title: "Other Reinforcement Learning Algorithms"
section_id: "14-02"
language: "en"
source_language: "zh"
source_docx: "第2部分 强化学习/14.强化学习的其他算法/14.2 基于模型的强化学习.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 14.2 Model-Based Reinforcement Learning

## I. Model Predictive Control (MPC)

Consider playing a board game. In a model-free algorithm, we observe the current board and place a piece directly from experience (the policy network directly outputs an action). MPC instead works as follows: after observing the board, I simulate in my mind (the environment model): if I play at A, my opponent might play at B, and then I can play at C... After simulating H steps, I find that the position is very favorable, so I decide to play at A as my first move now.

MPC's core is receding horizon control: from the current state $s_t$, first use the environment model to plan an action sequence for the next $H$ steps,

$$
a^*_{t:t+H-1}=\arg\max_{a_{t:t+H-1}}\sum_{\tau=t}^{t+H-1}r(s_\tau,a_\tau),
\quad s_{\tau+1}=f(s_\tau,a_\tau)
$$

Although an entire action sequence is optimized, only its first action $a_t^*$ is actually executed. The environment then reaches a new state $s_{t+1}$, from which the algorithm replans the next $H$ steps. In other words, MPC recalculates its plan while executing it, avoiding reliance on a long model prediction all at once.

Because the action space may be continuous and high-dimensional, directly solving the argmax above with a mathematical formula is difficult. We can therefore use the following approaches:

### (I) Shooting Method

A straightforward brute-force approach: randomly generate 1000 action trajectories, run them through the model, and select the one with the highest score. In high-dimensional spaces, random guessing rarely finds the optimum and is extremely inefficient.

### (II) Cross-Entropy Method (CEM)

1. Core steps

(1) Initialization: assume that actions follow a distribution (such as a normal distribution).

(2) Sampling: sample a batch of action sequences from this distribution.

(3) Evaluation: evaluate the sequences' scores using the model.

(4) Elite selection: select the highest-scoring sequences (such as the top 10%).

(5) Evolution: update the distribution's parameters with the elite sequences' statistics (mean and variance). Move the center toward high-scoring regions and reduce the variance (greater confidence).

(6) Iteration: repeat until the distribution converges.

2. Relationship to cross-entropy

CEM is an evolutionary algorithm. It generates many samples from the current distribution, selects the best-performing elite samples, and regards “these top 10% elite samples as the truth, the correct answer.” It then tends to update the distribution's parameters (such as u and sigma of a Gaussian) to maximize the probability of generating these “elite samples.” That is, the current distribution is fitted to the elite samples' distribution. Cross-entropy measures the difference between the two distributions, as derived below:

Let $z$ denote a candidate action sequence and $Q(z;\theta)$ the current sampling distribution. CEM does not aim to find an action directly; it seeks new distribution parameters $\theta_{\mathrm{new}}$ that make $Q(z;\theta_{\mathrm{new}})$ closer to the ideal distribution $P_{\mathrm{ideal}}(z)$ represented by the elite samples.

Use KL divergence to measure the difference:

$$
D_{\mathrm{KL}}(P_{\mathrm{ideal}}\Vert Q)
=\sum_z P_{\mathrm{ideal}}(z)\log\frac{P_{\mathrm{ideal}}(z)}{Q(z;\theta)}
$$

Expanding gives:

$$
D_{\mathrm{KL}}(P_{\mathrm{ideal}}\Vert Q)
=\sum_z P_{\mathrm{ideal}}(z)\log P_{\mathrm{ideal}}(z)
-\sum_z P_{\mathrm{ideal}}(z)\log Q(z;\theta)
$$

The first term depends only on $P_{\mathrm{ideal}}$ and not on $\theta$. Minimizing KL divergence is therefore equivalent to minimizing the cross-entropy term:

$$
\arg\min_\theta D_{\mathrm{KL}}(P_{\mathrm{ideal}}\Vert Q)
\Longleftrightarrow
\arg\min_\theta\left(-\sum_z P_{\mathrm{ideal}}(z)\log Q(z;\theta)\right)
$$

If the ideal distribution is understood as an empirical distribution with mass only on the elite samples, this is equivalent to maximizing their likelihood under the current distribution:

$$
\arg\max_\theta\sum_{z\in\mathrm{Elites}}\log Q(z;\theta)
$$

When $Q$ is Gaussian, this maximum likelihood estimate has an analytical update: the new mean is the elite samples' mean, and the new variance is their variance. Each CEM update therefore “re-estimates the sampling distribution from high-scoring samples,” concentrating the next round of samples in high-scoring regions.

## II. Probabilistic Ensembles with Trajectory Sampling (PETS) (UC Berkeley, NeurIPS 2018)

Models are imperfect. If the environment model predicts incorrectly, MPC plans an incorrect policy from incorrect information. PETS introduces probabilistic modeling and ensembles to quantify this risk.

### (I) Two Types of Uncertainty

1. Aleatoric uncertainty

This uncertainty comes from randomness in the environment itself and cannot be completely eliminated even by a well-learned model. Instead of outputting a single deterministic next state, PETS's environment model outputs the mean and variance of a Gaussian distribution:

$$
p_\theta(s_{t+1}\mid s_t,a_t)=N(\mu_\theta(s_t,a_t),\sigma_\theta^2(s_t,a_t))
$$

The corresponding negative log-likelihood loss can be written as:

$$
\mathcal L(\theta)=\sum_{i=1}^{N}-\log p_\theta(s_{t+1}^{(i)}\mid s_t^{(i)},a_t^{(i)})
$$

Expanding the Gaussian form and omitting constant terms gives the loss for one sample:

$$
\mathrm{loss}
=\frac{(s_{t+1}-\mu_\theta(s_t,a_t))^2}{2\sigma_\theta^2(s_t,a_t)}
+\log\sigma_\theta(s_t,a_t)
$$

The core point remains that PETS uses the mean and variance of its Gaussian output to model aleatoric uncertainty.

2. Epistemic uncertainty

This uncertainty comes from insufficient model knowledge. The less data the model has, and the fewer similar state-action pairs it has seen, the higher the uncertainty. PETS represents this using an ensemble:

- Train multiple environment models simultaneously, such as $M_1,\ldots,M_5$.
- Each model can have a different initialization and see slightly different data subsets through bootstrapping.
- If the models make substantially different predictions for the same action sequence, their knowledge in that region is insufficient.
- Similar predictions indicate that the region is more trustworthy.

### (II) PETS's Policy

PETS first collects a dataset of real interactions:

$$
\mathcal D=\{(s,a,r,s')\}
$$

It then initializes multiple probabilistic environment models:

$$
f_{\theta_1},f_{\theta_2},\ldots,f_{\theta_5}
$$

Each model takes $(s,a)$ as input and outputs Gaussian distribution parameters $N(\mu,\sigma)$ for the next state or state difference. During training, bootstrapping exposes different models to different data subsets, forming an ensemble.

The subsequent second and third stages use the models' outputs for forward computation. After several forward computations, the collected data are used to train the environment models, with the loss shown in subsection (III) below.

For planning, PETS combines MPC with CEM:

1. Initialize the action-sequence distribution.
2. Sample $N$ candidate action sequences from the distribution.
3. Generate $P$ particles for each candidate action sequence, such as $P=20$.
4. All particles start at the currently observed real state $s_0$.

The key to trajectory sampling is passing particles through different environment models. With 5 models $M_1,\ldots,M_5$, 20 particles can be assigned to models in groups: for example, particles 1–4 use $M_1$, particles 5–8 use $M_2$, and so on.

Each particle simulates several steps along the candidate action sequence and accumulates rewards. For a given candidate sequence, PETS averages the rewards of its 20 particles to obtain its score $A_i$:

$$
A_i=\frac{1}{P}\sum_{p=1}^{P}\sum_{\tau=t}^{t+H-1}r(s_{\tau}^{(p)},a_{\tau})
$$

It then performs CEM-style elite selection and distribution updates:

1. Rank candidate action sequences by $A_i$.
2. Select the top-scoring 10% as elite samples.
3. Re-estimate the sampling distribution's mean and variance from the elite action sequences.
4. Repeat sampling, evaluation, selection, and updating for several rounds.
5. Finally, take the first action corresponding to the updated distribution's mean as $a^*$.

The second and third stages are forward computation, namely “taking one step.” After a certain number of steps, the data collected during those steps are used to train the environment model.

Execution uses a receding horizon: execute only the first step of $a^*$, observe the real environment to obtain the next state $s_{t+1}$, add the new transition to $\mathcal D$, and then replan from the new real state or update the model periodically.

Intuitively, PETS resembles a Monte Carlo simulation: it constructs multiple “parallel simulations” for the same candidate action sequence. A sequence receives a high score only if most models and particles judge it favorably. If only one model is optimistic and the others disagree, the average score falls.

In other words, only actions with high overall scores and no major disagreement among environment models receive high A_i values, allowing them to enter the top 10% “elites” and be incorporated into the real action distribution. This policy is especially useful in environments with little tolerance for error.

### (III) Loss Function of the Environment Models (M1–M5)

Essentially, this maximizes the likelihood of correct predictions under the constraint of fitting a Gaussian distribution.

Let the model input be $x=(s_t,a_t)$ and the prediction target $y=s_{t+1}$ or a state difference. PETS models this with a Gaussian:

$$
p_\theta(y\mid x)=N(\mu_\theta(x),\sigma_\theta^2(x))
$$

Maximizing likelihood is equivalent to minimizing negative log-likelihood. The Gaussian probability density is:

$$
p_\theta(y\mid x)=\frac{1}{\sqrt{2\pi}\sigma_\theta(x)}
\exp\left(-\frac{(y-\mu_\theta(x))^2}{2\sigma_\theta^2(x)}\right)
$$

After removing constant terms, the loss is:

$$
L(\theta)=\frac{(y-\mu_\theta(x))^2}{2\sigma_\theta^2(x)}+\log\sigma_\theta(x)
$$

The first term is weighted mean squared error (MSE):

$$
\frac{(y-\mu_\theta(x))^2}{2\sigma_\theta^2(x)}
$$

If the model considers a region highly uncertain, meaning that $\sigma_\theta^2(x)$ is large, the same prediction error receives a smaller penalty. If the model is confident, meaning that $\sigma_\theta^2(x)$ is small, an incorrect prediction receives a larger penalty.

The second term is an uncertainty penalty:

$$
\log\sigma_\theta(x)
$$

It penalizes excessive uncertainty, preventing the model from increasing $\sigma_\theta(x)$ without bound to evade the weighted MSE. Overall, the model may increase variance appropriately in hard-to-predict regions but must pay the $\log\sigma$ penalty. In easy-to-predict regions, it should reduce variance to increase prediction confidence and lower the overall loss.

## III. Model-Based Policy Optimization (MBPO)

MBPO combines the high data efficiency of model-based methods with the high performance of model-free methods (such as SAC).

### (I) Compounding Errors in Model-Based Methods

Purely model-based methods (such as PETS) rely entirely on model simulations. If the first prediction is off by 1%, the second step predicts from already biased data, and errors accumulate exponentially. The longer the simulation, the less trustworthy the result.

### (II) MBPO's Innovation: Branched Rollouts

The traditional approach simulates with the model from s_0 all the way to the end, but MBPO does not. Its steps are as follows:

MBPO's branched rollouts start from states in the real replay buffer:

1. The real environment generates transitions $(s,a,r,s')$, which are stored in real replay buffer $\mathcal D_{\mathrm{env}}$.
2. Sample real state $s_{\mathrm{real}}$ from $\mathcal D_{\mathrm{env}}$.
3. Generate an action at $s_{\mathrm{real}}$ with the current policy.
4. Generate a short rollout with the learned environment model.
5. Store model-generated transitions in model replay buffer $\mathcal D_{\mathrm{model}}$.
6. Train a model-free policy, such as SAC, on a mixture of data from the real and model replay buffers.

Each rollout starts from a “real state,” like checking the map (real data) every few steps, eliminating accumulated errors. Although each rollout is short, thousands of such short data sequences can be generated. For algorithms such as SAC, more data lead to more stable training. MBPO demonstrates that if the model rollout length k is controlled appropriately (balancing model error and data gains), model-based methods can be many times faster than model-free methods (more sample-efficient), with equally good final performance.

### (III) MBPO Workflow

Initialization and warm-up

Initialization includes:

1. Initialize the Actor and Critic, namely SAC's policy and value networks.
2. Initialize the ensemble dynamics model.
3. Initialize real replay buffer $\mathcal D_{\mathrm{env}}$ and model replay buffer $\mathcal D_{\mathrm{model}}$.
4. Warm up for several steps with a random policy, first filling $\mathcal D_{\mathrm{env}}$ with real interaction data.

Main loop:

Overall, MBPO inherits PETS's ensemble environment models (M1–M5) and their update process (the second stage). However, instead of using the environment model to determine the policy directly as PETS does, MBPO works like Dyna-Q: select a previously observed real state, generate an action with the current policy, and let the environment model generate auxiliary training data (the third stage). The main algorithm then updates using both real and model data (the fourth stage).

Stage 1: real interaction

The current policy interacts with the real environment for one step:

1. In real state $s_{\mathrm{real}}$, sample action $a_t$ with the current policy.
2. The real environment returns reward $r_t$ and the next real state $s'_{\mathrm{real}}$.
3. Store $(s_{\mathrm{real}},a_t,r_t,s'_{\mathrm{real}})$ in real replay buffer $\mathcal D_{\mathrm{env}}$.
4. Return to the main loop and continue collecting real data.

Stage 2: environment model updates (feed the latest reality to the environment model and update ensemble environment models M1–M5, as in PETS, using exactly the same loss as PETS)

The environment model update stage:

1. Sample a minibatch from real replay buffer $\mathcal D_{\mathrm{env}}$.
2. Train the ensemble by maximum likelihood or negative log-likelihood.
3. The model input is $(s,a)$; the target is to predict the distribution $N(\mu,\sigma)$ of the next state and reward.
4. A full retraining is not needed at every step. In practice, train periodically after a certain number of real interaction steps, for example once every 250 steps.

Stage 3: branched rollouts (let the environment model output simulated data to feed into SAC)

The branched rollout stage:

1. Sample real state $s_{\mathrm{real}}$ from $\mathcal D_{\mathrm{env}}$ as the rollout starting point.
2. Simulate only a short horizon of $k$ steps, such as $k=1$ or another small integer.
3. At each step, first obtain an action from the current policy, then predict the next state and reward with the ensemble model.
4. Store synthetic model-generated transitions in $\mathcal D_{\mathrm{model}}$.
5. Short rollouts control accumulated model error while still greatly increasing the data available for policy training.

This stage differs from Dyna-Q: Dyna-Q directly takes (s_t,a_t) and predicts only s_t+1,r_t. Here, only s_real is taken, and the subsequent rollout follows the current policy.

Stage 4: policy optimization (train SAC with real data and environment model data)

The policy optimization stage trains SAC:

1. **Mixed sampling**: sample from the two buffers in a certain proportion to construct a training batch.
   - Draw a small portion from $\mathcal D_{\mathrm{env}}$ to maintain realism.
   - Draw the vast majority from $\mathcal D_{\mathrm{model}}$ to use the model's efficiency.
   - For example: Batch Size = 256, with 10% real data and 90% model data.
2. **SAC update**: update the Actor and Critic parameters using this mixed batch (gradient descent).
   - Note: MBPO's characteristic is “one environment step, many network training steps.” Typically, SAC performs $G$ gradient updates (such as 20) for every 1 real environment step. This greatly improves sample utilization.
3. This uses model data to improve sample efficiency while retaining SAC's stable optimization in continuous-control tasks.

The fourth stage is similar to combining Dyna-Q's “update using the latest real interaction” and “update using the environment,” enabling better parallel computation.

Pseudocode:

```text
Initialize;

First play randomly for 5000 steps and store them in the real buffer;

for epoch in range(total_epochs):

--- Step A: Real interaction ---

Execute one interaction step following SAC's policy;

Store it in the real replay buffer;

--- Step B: Train/update the model ---

if epoch % 250 == 0: # Retrain the model every 250 real interaction steps

Train the model on real data to minimize Negative Log-Likelihood;

--- Step C: Branched rollout (data generation) ---

Select starting points from the real buffer;

for i in range(k_steps): # k-step rollout

Generate an action with the current policy and randomly choose a model network to predict the next state and reward;

Store it in the model replay buffer;

Update the state to prepare for the next rollout step;

--- Step D: Policy update (SAC) ---

for _ in range(G_updates): # Train SAC G times (such as 20) per environment step

Sample a mixture (usually with a high proportion of model data);

Update SAC parameters;
```

## References

- [Hands-on Reinforcement Learning (translated title; in Chinese)](https://hrl.boyuai.com/).
- Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114). NeurIPS 2018.
- Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253). NeurIPS 2019.
