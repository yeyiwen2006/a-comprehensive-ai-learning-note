---
title: "13.3 PPO and GAE"
chapter_title: "Combining Value and Policy Methods"
section_id: "13-03"
language: en
source_language: zh
source_docx: "第2部分 强化学习/13.综合价值与策略的算法/13.3 PPO算法与GAE.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 13.3 PPO and GAE

PPO derives from TRPO. TRPO introduced importance sampling and restricted update regions, but solving its constrained optimization problem is complex. It is “precise but expensive” and impractical for large models. Constraints can instead be incorporated directly into the objective. Two ways to do this produce two PPO variants.

## I. PPO-Penalty

The objective is:

$$
\begin{aligned}
J^{\mathrm{PEN}}(\theta)
&=
\mathbb{E}_t\left[
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
A_t
-
\beta\cdot
D_{\mathrm{KL}}\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\Vert
\pi_\theta(\cdot\mid s_t)
\right)
\right]
\end{aligned}
$$

* First term: TRPO's surrogate objective, Ratio $\times$ Advantage, improving the policy.
* Second term: a KL penalty, with $D_{\mathrm{KL}}$ measuring the difference between old and new policy distributions.
* $\beta$: the penalty coefficient. Larger $\beta$ permits less dramatic policy change.

The first term is the same as TRPO's: find a new policy maximizing expected reward under that policy. The forward KL term chiefly ensures that regions with high old-policy probability $\pi_{\mathrm{old}}$, such as common human language patterns and basic world knowledge learned in pretraining, retain high probability under $\pi_{\mathrm{new}}$. This prevents discarding pretrained knowledge to gain reward and causing “mode collapse.” Meanwhile, increasing probabilities of difficult solutions that had low old-policy probability incurs no excessive penalty. This ensures updates remain within a credible “trust region.”

PPO-Penalty's essence is dynamically adjusting $\beta$ from actual KL after each update rather than fixing it. Let:

$$
d = D_{\mathrm{KL}}(\pi_{\mathrm{old}},\pi_{\mathrm{new}})
$$

Set target KL $d_{\mathrm{targ}}$, such as $0.01$:

1. If $d < d_{\mathrm{targ}}/1.5$, the change is too small and conservative, so reduce $\beta$, for example:

$$
\beta \leftarrow \beta/2
$$

This allows a larger next step.

2. If $d > d_{\mathrm{targ}}\times 1.5$, the change is too large and aggressive, so increase $\beta$, for example:

$$
\beta \leftarrow 2\beta
$$

This penalizes the next update more heavily.

## II. PPO-Clip (Mainstream)

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

$$
\begin{aligned}
J^{\mathrm{CLIP}}(\theta)
&=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)A_t,\,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
\end{aligned}
$$

$A_t$ is the action's relative value (discussed below). We seek a new policy maximizing expected value; importance sampling yields the expression above. The derivative of $J$ with respect to $\theta$ is the parameter-network update gradient. When $A_t$ is greater than $0$, we want larger $\pi_\theta$, equivalently larger $r_t$. $E_t$ denotes data sampled under the old policy.

In practice, first run several steps under $\pi_{\mathrm{old}}$, then perform several updates. Throughout each round, keep $\pi_{\mathrm{old}}$ fixed as the data-collection policy (implemented as a direct arithmetic average over past samples), repeating:

When $A_t>0$, the action is good:

* Originally, larger $r_t$ is preferred, increasing the action probability.
* Clipping: once $r_t>1+\epsilon$, this term becomes constant $(1+\epsilon)A_t$, with derivative $0$.
* Result: policy updating stops, preventing an accidentally high score from raising the action probability without limit.

When $A_t<0$, the action is bad:

* Originally, smaller $r_t$ is preferred, reducing the action probability.
* Clipping: once $r_t<1-\epsilon$, the term becomes $(1-\epsilon)A_t$, with derivative $0$.
* Result: policy updating stops, preventing one negatively rewarded action from excessively disrupting policy structure.

Once outside the range, $J$ is “flattened,” and the current update seeking to maximize it “loses motivation” and naturally stops.

## III. Why Does PPO-Clip Perform Better?

1. Stability: although PPO-Penalty adjusts $\beta$ dynamically, delayed adjustments or oscillation can destabilize training. PPO-Clip applies hard clipping to individual samples and is extremely robust.

2. Computation: PPO-Penalty calculates KL between $\pi_{\mathrm{new}}$ and $\pi_{\mathrm{old}}$ (simpler than TRPO's second derivatives but still requiring log-probability differences). Both probabilities change, making computation expensive. PPO-Clip needs only multiplication and comparisons (min, clamp), reducing cost.

3. Hyperparameters: fixing PPO-Clip's $\epsilon$ at $0.2$ usually suits most environments (Atari, MuJoCo, robots), whereas PPO-Penalty is more sensitive to $d_{\mathrm{targ}}$ and initial $\beta$.

## IV. Loss Function

$$
L_{\mathrm{total}}=-\underbrace{L^{\mathrm{CLIP}}}_{\text{Actor loss}}+c_1\underbrace{L^{\mathrm{VF}}}_{\text{Critic loss}}-c_2\underbrace{S}_{\text{Entropy}}
$$

Note the subtle signs: the optimizer is assumed to perform gradient descent, minimizing the objective.

* $L^{\mathrm{CLIP}}$: reward should be maximized, so gradient descent requires the negative, $-L^{\mathrm{CLIP}}$.
* $L^{\mathrm{VF}}$: prediction error (MSE) should be minimized, so use $+L^{\mathrm{VF}}$ directly.
* $S$: entropy should be maximized to encourage exploration, so use $-S$ for gradient descent.

Adding them lets a PyTorch/TensorFlow optimizer optimize all objectives together.

During backpropagation, different loss terms send gradients to corresponding variables in different modules:

```text
Input image --> [Shared CNN layers] --> Extracted feature vector
                                  |
                                  +-----------------------------+
                                  |                             |
                             [Actor head]                  [Critic head]
                         Action probabilities               Value V
```

## V. Advantage $A_t$: Generalized Advantage Estimation (GAE)

Advantage $A_t$ measures how much better action $a_t$ is than “average performance” in that state. The theoretical definition is:

$$
A_t=Q(s_t,a_t)-V(s_t)
$$

In practice, $Q$ is unknown and must be estimated from samples.

Case 1: one-step estimation

Looking only one step ahead, action $a_t$'s $Q$ value is estimated by:

$$
r_t+\gamma V(s_{t+1})
$$

The estimated $A_t$ then equals TD error:

$$
\hat{A}_t^{(1)}=\underbrace{r_t+\gamma V(s_{t+1})}_{\approx Q(s_t,a_t)}-V(s_t)=\delta_t
$$

PPO uses GAE by default to estimate advantages. This ingenious design takes an exponentially weighted moving average of TD errors:

$$
\hat{A}_t^{\mathrm{GAE}}
=
\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^{2}\delta_{t+2}+\cdots
$$

Here $\lambda$ is a hyperparameter in $[0,1]$.

$\lambda$ controls the influence of TD errors:

For $\lambda=0$, bias is high and variance low:

$$
\hat{A}_t=\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$A_t$ is exactly the current one-step TD error.

* Advantage: low variance, depending only on one random reward $r_t$.
* Disadvantage: high bias, heavily dependent on critic accuracy $V(s_{t+1})$. A poorly trained critic gives incorrect estimates.

For $\lambda=1$, there is no bias but high variance:

$$
\hat{A}_t=\sum_{l=0}^{\infty}\gamma^{l}\delta_{t+l}=\left(\sum_{l=0}^{\infty}\gamma^{l}r_{t+l}\right)-V(s_t)
$$

Expanding cancels intermediate $V$ terms, leaving Monte Carlo returns (actual returns) minus the baseline.

* Advantage: low bias; actual return is the most accurate “fact.”
* Disadvantage: extremely high variance from accumulating environmental randomness at every step.

Which actions produce $A_t$? As discussed later, PPO first runs several steps under the old policy, computes advantages for all visited times $t$, and then updates. Thus $A_t$ is computed from the actions actually taken during those steps.

## VI. Practical Workflow (Such as OpenAI Baselines or Stable Baselines3)

Updating after each step means serial computation and extremely low GPU utilization. Since importance sampling ensures data reuse, we usually work in a “large loop”: run several steps at once (such as 2048), collect a batch, train and update the policy on it, then discard it.

Initialization: initialize actor $\pi_\theta$ and critic $V_\phi$.

Each iteration has four phases:

**1. Data collection through interaction**

* Run current policy $\pi_{\theta_{\mathrm{old}}}$ in the environment for $T$ steps, such as 2048.
* Collect trajectory data:

$$
\{s_t,a_t,r_t,s_{t+1},\log\pi_{\mathrm{old}}(a_t\mid s_t)\}
$$

* Use the critic to compute state values $V(s_t)$.

**2. Advantage computation**

* Compute TD errors:

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

* Compute GAE $\hat{A}_t$, a recursive formula balancing bias and variance.

Note: values $V(s_t)$ are usually computed during collection because PPO actors and critics often share lower network layers (CNN/MLP). Choosing $a_t$ already requires feeding $s_t$ through the network, so obtaining the critic's $V(s_t)$ costs almost nothing extra. Without storing it, training would need another pass of $s_t$ to obtain $V(s_t)$ for calculating advantages, duplicating computation and wasting GPU resources.

Note: advantages are computed together after the action steps and before policy-gradient updates. GAE weights TD errors and accumulates them from t through 2048.

**3. Optimization**

* Important: $\pi_{\theta_{\mathrm{old}}}$ is fixed as the denominator; new $\pi_\theta$, the numerator, is optimized.
* Shuffle the $T$ collected samples into minibatches, such as 64 samples each.
* Repeat the epoch loop, for example 10 times.

For each minibatch:

* Compute the new probability ratio:

$$
r_t(\theta)=\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}
$$

* Compute clipped loss $L^{\mathrm{CLIP}}$.
* Compute critic value loss:

$$
L^{\mathrm{VF}}=\left(V_\phi(s_t)-V_{\mathrm{target}}\right)^{2}
$$

* Compute entropy regularizer $S$ to encourage exploration.
* Compute total loss:

$$
L=-L^{\mathrm{CLIP}}+c_1L^{\mathrm{VF}}-c_2S
$$

* Backpropagate and update $\theta$ and $\phi$.

**4. Synchronize the policy**

* After this round, set:

$$
\pi_{\theta_{\mathrm{old}}}\leftarrow\pi_\theta
$$

* Clear the data buffer and start the next large loop.

Suppose parameters are:

* `n_steps`, buffer size: $2048$.
* `batch_size`, minibatch size: $64$.
* `n_epochs`, reuse count: $10$.

The number of backpropagations per “large loop” is:

1. Batching: split 2048 samples into:

$$
2048/64=32
$$

minibatches.

2. One pass: sequentially feed these 32 minibatches to the GPU, producing 32 backpropagations.
3. Repeated reuse: run 10 epochs.
4. Total:

$$
32\times 10=320
$$

Thus each large loop produces 320 backpropagations.

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). ICLR 2016.
