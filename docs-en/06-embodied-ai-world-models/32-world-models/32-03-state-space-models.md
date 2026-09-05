---
title: "32.3 State-Space Models"
chapter_title: "World Models"
section_id: "32-03"
language: en
source_language: zh
source_docx: "第6部分 具身智能与世界模型/32.世界模型/32.3 状态空间模型.docx"
status: "manually rebuilt and checked against Word"
ocr: "all Word-visible text and formula images manually transcribed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 32.3 State-Space Models

## I. Recurrent State-Space Model (RSSM)

### (1) Model Architecture

There are usually two extremes when constructing an environmental dynamics model:

1. **Purely deterministic models (such as standard RNNs)**: State transitions are fully deterministic. A critical weakness is their inability to capture inherent uncertainty in environmental dynamics and multiple possible futures. When used for planning, the planner can easily exploit their inaccuracies (that is, overfit to a single incorrect future).

2. **Purely stochastic models (such as traditional SSMs)**: Every state transition includes sampling from a probability distribution. Although this models uncertainty, purely stochastic transitions make it extremely difficult to reliably remember information across multiple time steps.

Moreover, when the environment is stochastic, deterministic networks often output a “blurred average” of all possible futures.

RSSM uses an LSTM-like approach, separating the hidden state (historical memory) from the latent representing the current state to combine the advantages of both. In a partially observable Markov decision process (POMDP):

Historical memory $h_t$ serves as the hidden state and transitions deterministically: it is obtained by applying a function to $h_{t-1}$, the memory of time $t-2$ and earlier, together with the state latent $s_{t-1}$ at time $t-1$ and action $a_{t-1}$ at time $t-1$.

1. **Deterministic state model**

$$
h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
$$

- **Physical meaning**: Summarize historical information and maintain deterministic memory across multiple time steps.
- **Fitting method**: Usually fitted by a recurrent neural network (RNN, such as a GRU).

The world-state latent $s_t$ is predicted from historical memory $h_t$, which contains states and actions at time $t-1$ and earlier, through random sampling:

2. **Stochastic state model / prior**

$$
s_t \sim p(s_t \mid h_t)
$$

- **Physical meaning**: Predict the distribution of possible current states solely from past memory, without observing the current real image. This introduces uncertainty to describe multiple possible futures.
- **Fitting method**: A multilayer perceptron (MLP) outputs the mean and variance of a diagonal Gaussian distribution.

The actual world-state latent $s_t$ (the “correct prediction answer” after obtaining current real observation $o_t$) is represented as follows. (Note: Bayesian computation is theoretically possible, but it involves intractable integrals, so a neural network is fitted directly.)

3. **Approximate posterior / variational encoder**

$$
q(s_t \mid h_t, o_t)
$$

- **Physical meaning**: Combine past deterministic memory $h_t$ with the current real high-dimensional pixel observation $o_t$ to infer the most accurate current latent-state distribution.
- **Fitting method**: A convolutional neural network (CNN) extracts image features, followed by an MLP that outputs the mean and variance of a diagonal Gaussian distribution.

After predicting $s_t$, we can generate the pixel image $o_t$ from $h_t$ and $s_t$:

4. **Observation reconstruction model / decoder**

$$
o_t \sim p(o_t \mid h_t, s_t)
$$

- **Physical meaning**: Verify that the latent state contains enough environmental information by reconstructing a high-dimensional pixel image from it.
- **Fitting method**: Fitted by a deconvolutional neural network.

And predict the reward:

5. **Reward prediction model**

$$
r_t \sim p(r_t \mid h_t, s_t)
$$

- **Physical meaning**: Predict the scalar reward available in the current latent state, which is the planner's sole basis for evaluating actions in latent space.
- **Fitting method**: Fitted by an MLP.

### (2) Workflow

- **Step 1: Data sampling**

Uniformly sample sequence segments $`\lbrace(o_t, a_t, r_t)_{t=k}^{L+k}\rbrace_{i=1}^{B}`$ of length $L$ with batch size $B$ from replay buffer $\mathcal{D}$.

- **Step 2: Forward pass for state unrolling and distribution computation**

For each time step $t = 1 \ldots L$ in the sequence:

1. Perform the deterministic state transition: Feed the previous $h_{t-1}$, $s_{t-1}$ (sampled through reparameterization), and action $a_{t-1}$ into RNN $f$ to obtain the current deterministic state $h_t$.
2. Compute prior distribution $p$: Feed $h_t$ into an MLP to output the current prior Gaussian's mean and variance, $\mu_{\mathrm{prior}}$ and $\sigma_{\mathrm{prior}}$.
3. Compute posterior distribution $q$: Extract features from current image $o_t$ with a CNN and feed them, together with $h_t$, into another MLP to output the posterior Gaussian's mean and variance, $\mu_{\mathrm{post}}$ and $\sigma_{\mathrm{post}}$.
4. Reparameterized sampling: Sample latent vector $s_t$ from posterior $\mathcal{N}(\mu_{\mathrm{post}}, \sigma_{\mathrm{post}}^2)$.

- **Step 3: Reconstruction and prediction**

Feed $h_t$ and sampled $s_t$ into the deconvolutional decoder to predict reconstructed image $\hat{o}_t$, and into the reward model to predict scalar reward $\hat{r}_t$.

- **Step 4: Loss computation**

1. Compute the MSE between reconstructed image $\hat{o}_t$ and true image $o_t$ (reconstruction loss).
2. Compute the MSE between predicted reward $\hat{r}_t$ and true reward $r_t$.
3. Compute the KL divergence between prior $p$ and posterior $q$ (when using latent overshooting, also compute and accumulate in parallel the KL divergences between multistep priors and current posteriors).

- **Step 5: Parameter update through backpropagation**

Sum and average losses across all time steps to obtain total loss $\mathcal{L}(\theta)$. Use an optimizer (such as Adam) to compute gradients and update all neural-network parameters constituting these five functions: $\theta \leftarrow \theta - \alpha \nabla_{\theta}\mathcal{L}(\theta)$.

### (3) Loss Function

The loss has three components. The KL divergence between the current-latent prediction p based only on past memory and the true current latent q based on past memory and the current observation expresses the discrepancy between predicted and true next-step latents, reflecting next-state prediction ability. The image-reconstruction loss reflects the ability to reconstruct pixels from latents, while the reward loss reflects the ability to predict rewards from latents. The expression is:

$$
\begin{aligned}
\mathcal{L}
&=
\sum_{t=1}^{T}
\Biggl(
-\mathbb{E}_{q(s_t \mid o_{\le t}, a_{<t})}
\left[\ln p(o_t \mid s_t) + \ln p(r_t \mid s_t)\right]
\\
&\quad+
\mathbb{E}
\left[
D_{\mathrm{KL}}\left(
q(s_t \mid o_{\le t}, a_{<t})
\parallel
p(s_t \mid s_{t-1}, a_{t-1})
\right)
\right]
\Biggr)
\end{aligned}
$$

Here, both $o_t$ and $r_t$ follow Gaussian distributions, so these two terms are equivalent to MSE losses.

Because the standard objective trains stochastic paths only through one-step KL divergence, the model can diverge during multistep prediction. Latent overshooting extends the constraint to all future time steps at distance $d$:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{overshooting}}(\theta)
&=
\sum_{t=1}^{T}
\Biggl(
-\mathbb{E}_{q}\left[\ln p(o_t \mid s_t)\right]
\\
&\quad+
\frac{1}{D}
\sum_{d=1}^{D}
\beta_d
\mathbb{E}
\left[
D_{\mathrm{KL}}\left(
q(s_t \mid o_{\le t})
\parallel
p(s_t \mid s_{t-d})
\right)
\right]
\Biggr)
\end{aligned}
$$

- **Physical meaning**: Not only must “the prior for step $t$ predicted from step $t - 1$” align with the posterior, but so must “the prior for step $t$ obtained by consecutive multistep rollouts from step $t - 2$ or $t - d$.” This latent-space consistency regularization strengthens long-term evolution without relying on real observations.

## II. RSSM-Based Applications

### (1) PlaNet: Search-Based Planning with a World Model

1. **Outer loop: Model training and data collection (Algorithm 1)**

- **Initialization**: Collect a small number of initial seed episodes with random actions and store them in replay buffer $\mathcal{D}$.
- **Model fitting**: Randomly sample data chunks from $\mathcal{D}$, compute the loss using the variational-lower-bound formula above, and update RSSM parameters $\theta$ through gradient ascent.
- **Environmental interaction**: Use the updated model to choose action $a_t$ through the planning algorithm (CEM below), add Gaussian exploration noise to the action, interact with the environment, and store the new experience in $\mathcal{D}$.

2. **Inner loop: CEM-Based Action Planning (Algorithm 2)**

At every time step, instead of using a policy network to output actions, the agent searches for the optimal action sequence in real time using the cross-entropy method (CEM) and model predictive control (MPC):

- **Initialize the belief**: Construct a time-varying Gaussian belief $q(a_{t:t+H}) \sim Normal(\mu, \sigma^2 I)$ representing the best action sequence over planning horizon $H$.
- **Iterative optimization**:
  1. Sample $J$ candidate action sequences from the current belief.
  2. Feed them into the learned RSSM, roll out future states only in latent space, and compute total expected rewards $R^{(j)}$.
  3. Select the $K$ highest-reward sequences.
  4. Refit the action-belief distribution with the mean and variance of these $K$ elite sequences.
- **Execute and replan**: After $I$ iterations, execute the first action of the final belief mean. Because MPC is used, the belief over action sequences is reset to zero mean and unit variance after a new observation arrives, avoiding local optima.

### (2) Dreamer 3: RL Training with a World Model

**1. World Model**

- The world model uses autoencoding to compress complex perceptual inputs (such as images) into compact latent representations.
- It is implemented as a recurrent state-space model (RSSM).
- First, the encoder maps perceptual input $x_t$ to stochastic representation $z_t \sim q_{\phi}(z_t \mid h_t, x_t)$.
- Then, the dynamics predictor predicts future representations from the previous state $h_t$:

$$
\hat{z}_t \sim p_{\phi}(\hat{z}_t \mid h_t).
$$

- In this way, the world model learns the environment's intrinsic structure and can “imagine” future state evolution in latent space.

Mathematically, RSSM is a generative model combining a deterministic RNN and stochastic state variables. At time step $t$, it defines action $a_t$, observation input $x_t$, reward $r_t$, and an episode-continuation flag $c_t \in \{0, 1\}$.

**2. Critic**

**Step 2: Critic Learning**

- The critic and actor learn entirely from abstract trajectories “imagined” by the world model, without directly interacting with the real environment.
- The critic receives model state $s_t = [h_t, z_t]$ and learns to evaluate its value.
- To account for long-term returns, the critic predicts the expected value of the return distribution and uses bootstrapped $\lambda$-returns to integrate predicted rewards and values.

**3. Actor**

**Step 3: Actor Learning**

- The actor aims to select return-maximizing actions while encouraging exploration through entropy regularization.
- It is optimized on imagined trajectories with a policy-gradient method, the REINFORCE estimator.

**4. Loss Functions**

(1) Prediction loss

$$
\mathcal{L}_{\mathrm{pred}}(\phi)
\doteq
-\ln p_{\phi}(x_t \mid z_t, h_t)
-\ln p_{\phi}(r_t \mid z_t, h_t)
-\ln p_{\phi}(c_t \mid z_t, h_t)
$$

Through negative log-likelihood, this component forces the model to accurately reconstruct observation inputs, rewards, and episode flags.

(2) Dynamics loss

$$
\mathcal{L}_{\mathrm{dyn}}(\phi)
\doteq
\max\left(
1,
D_{\mathrm{KL}}\left[
\mathrm{sg}(q_{\phi}(z_t \mid h_t, x_t))
\parallel
p_{\phi}(\hat{z}_t \mid h_t)
\right]
\right)
$$

Its main purpose is to make dynamics predictor $p_{\phi}$ approximate the encoder's posterior $q_{\phi}$. The stop-gradient operation $\mathrm{sg}(\cdot)$ means that bringing the distributions closer updates only the dynamics predictor's parameters, without affecting how the encoder extracts features.

The outer $\max(1, \cdot)$ also introduces free bits: when KL divergence falls below 1 nat $\approx 1.44$ bits, the model stops penalizing this loss, avoiding overly degenerate, trivial dynamics and shifting learning back toward prediction loss.

(3) Representation loss

$$
\mathcal{L}_{\mathrm{rep}}(\phi)
\doteq
\max\left(
1,
D_{\mathrm{KL}}\left[
q_{\phi}(z_t \mid h_t, x_t)
\parallel
\mathrm{sg}(p_{\phi}(\hat{z}_t \mid h_t))
\right]
\right)
$$

Unlike the preceding loss, representation-loss gradients are restricted to encoder $q_{\phi}$. The aim is to constrain the encoder in the opposite direction, encouraging it to extract more “regular” hidden states that the dynamics model can predict more easily.

Through careful use of stop-gradient operations and different loss scales (set in the paper to $\beta_{\mathrm{pred}} = 1$, $\beta_{\mathrm{dyn}} = 1$, and $\beta_{\mathrm{rep}} = 0.1$), RSSM resolves the previous dilemma that “strong regularization works on simple images but erases critical details in complex 3D environments,” achieving cross-domain robustness.

## III. Dreamer 4: Decoupling Reconstruction, Dynamics-Prediction, and Reward-Prediction Losses

Stepwise loss decoupling is an improvement introduced by Google DeepMind in its world model Dreamer 4.

### (1) Background

In Dreamer 3, the image encoder, dynamics model (RSSM), image decoder, and reward predictor are trained simultaneously as one large whole.

During every backpropagation pass, latent features $z_t$ are pulled by three forces:

1. **Reconstruction gradients** ($\nabla \mathcal{L}_{recp}$): Force $z_t$ to remember every cloud and tree in the scene because the decoder must reconstruct the entire image.
2. **Dynamics gradients** ($\nabla \mathcal{L}_{dyn}$): Force $z_t$ to focus on how objects move.
3. **Reward-prediction gradients** ($\nabla \mathcal{L}_{pred}$): Force $z_t$ to focus entirely on tiny score-relevant objects, such as distant monsters or diamonds held in hand, while ignoring backgrounds that currently yield no points but are crucial to understanding the physical world.

The critical problem is “competition for representational capacity.” In very complex environments (such as Minecraft), these three gradients conflict. If reward signals are strong, the encoder discards background details in $z_t$ that “currently yield no reward but are crucial to understanding the physical world,” such as terrain variation, to satisfy reward prediction. This makes Dreamer 3's world model highly susceptible to representation collapse in complex 3D environments.

### (2) Components of the Decoupled Losses

1. **Tokenizer Compression Loss**

Unlike its predecessor, Dreamer 4 no longer reconstructs images while rolling out dynamics in latent space. It first trains a causal tokenizer using masked-autoencoder (MAE) logic: video patches are randomly masked with probability $p \sim U(0, 0.9)$, focusing exclusively on maximal spatiotemporal-feature compression.

Its loss abandons Symlog in favor of the standard combination of reconstruction and perceptual losses from pure vision:

$$
\mathcal{L}_{\mathrm{tokenizer}}=\mathrm{MSE}(x, \hat{x})+\mathrm{LPIPS}(x, \hat{x})
$$

- MSE ensures pixel-level accuracy.
- Learned perceptual image patch similarity (LPIPS) ensures structural integrity in human visual perception, avoiding blurry generated features.

2. **Dynamics-Prediction Loss**

This term measures the Transformer's next-step prediction accuracy. Dreamer 4 abandons Dreamer 3's KL divergence to avoid mode collapse when handling long sequences, replacing it with flow matching and shortcut forcing.

3. **Task and Reward-Prediction Loss**

Dreamer 4 trains the actor and critic within the world model, which must output reward and action predictions:

After training the tokenizer and dynamics on vast quantities of unlabeled videos, Dreamer 4 ingests labeled data, such as Minecraft recordings with mouse and keyboard actions, to train an extremely lightweight task head. This loss resembles Dreamer 3's $\mathcal{L}_{pred}$ but adds cloning-based prediction of specific actions:

$$
\mathcal{L}_{\mathrm{task}}=-\sum_{n=0}^{L}\ln p_{\theta}(r_{t+n}\mid h_t)-\sum_{n=0}^{L}\ln p_{\theta}(a_{t+n}\mid h_t)
$$

The model uses historical spatiotemporal context $h_t$ aggregated by the Transformer to simultaneously maximize the log-likelihoods of future reward sequences $r_{t+n}$ and true action sequences $a_{t+n}$.

### (3) Benefits of Loss Decoupling

1. **Avoiding Gradient Conflicts**

- **Step One (visual independence)**: First train the tokenizer to its full capability using only reconstruction loss (MAE-like autoencoder logic). At this point, $z_t$ is a perfect **purely visual, objective representation**, entirely uncontaminated by reward signals. Freeze the tokenizer after training.
- **Step Two (physical independence)**: Separately train a block-causal Transformer (the dynamics model) on $z_t$ extracted by the frozen tokenizer. The model cares only about how one frame becomes the next (shortcut forcing) and knows nothing about “rewards.” After training, the world model's physical-rollout capability is also frozen (or trained at a low learning rate).
- **Step Three (task alignment)**: Finally, add a very small network on top of this frozen large model that understands vision and physics, fitting specific rewards $r_t$ and actions $a_t$.

2. **Data Scalability**

End-to-end reinforcement learning such as Dreamer 3 requires every data frame to contain an exact (state, action, reward, next state) tuple; otherwise, the missing reward-prediction-branch gradients would bias learning. In the real world, however, obtaining vast amounts of action- and reward-labeled data is extremely expensive and difficult. Through decoupling, Dreamer 4's first two and largest stages (the tokenizer and flow-matching Transformer) require neither reward nor action labels and can learn directly from vast quantities of unlabeled videos.

## References

- Hafner, D., Lillicrap, T., Fischer, I., et al. (2019). [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551). ICML. (PlaNet; the paper introduces RSSM)
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). arXiv:2301.04104. (Dreamer 3)
- Hafner, D., Yan, W., & Lillicrap, T. (2025). [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527). arXiv:2509.24527. (Dreamer 4)
