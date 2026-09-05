---
title: "27.3 Reinforcement Learning in Flow-Matching Models and FlowGRPO"
chapter_title: "Flow-Matching Models"
section_id: "27-03"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/27.流匹配模型/27.3 流匹配模型中的强化学习与FlowGRPO.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 27.3 Reinforcement Learning in Flow-Matching Models and FlowGRPO

## I. Changes to the Markov Process

In flow-matching models, the Markov process differs from that in purely autoregressive LLMs:

1. **State $x_t$**: the continuous image or latent variable at the current time step.
2. **Action $v_t$**: the continuous vector field output by the model, that is, a direction.
3. **Environment transition**: deterministic numerical integration, such as Euler's method $x_{t+\Delta t}=x_t+v_t\Delta t$.
4. **Reward**: an aesthetic or alignment score $R(x_1)$ is available only after integration to $t=1$ produces the final image $x_1$.

The process from $x_0$ to $x_1$ is essentially the integral of $v_t$ with respect to $t$ from 0 to 1, where $v_t$ is continuous. In practice, it is generally sampled over several dozen steps, moving a small distance along the velocity at each time step.

## II. Specific Difficulties of RL in Flow Matching

Two difficulties of applying RL to diffusion models were discussed earlier: credit assignment over long trajectories with sparse rewards, and the exploration crisis in extremely high-dimensional continuous action spaces. Flow-matching models also face unique RL difficulties because of their underlying mathematical assumptions.

### 1. Breaking the “straight-line property” of optimal transport and leaving the manifold

- **The difficulty**: the main advantage of flow matching during pretraining, particularly today's mainstream OT-FM, is that it uses optimal transport to force alignment between pure noise $x_1$ and real images $x_0$, constructing a very simple straight trajectory with zero curvature:

$$
x_t=t x_1+(1-t)x_0
$$

- **The straight line breaks**: pretrained FM is a function approximator that perfectly follows this line. When RL is introduced, the sole objective of its policy gradient is to “maximize the final image's reward.” To satisfy the reward, RL forcibly modifies $v_t$ at intermediate time steps, inevitably “pulling” the ODE trajectory away from the original optimal-transport line.
- **Consequence**: diffusion-model trajectories are already curved, high-curvature Brownian motion, so some additional deviation is tolerable. For FM, however, leaving the straight line not only produces off-manifold, meaningless images; more critically, the sharp increase in curvature causes the ODE solver's truncation error to explode. FM may originally produce an image in only 4 integration steps, but after RL distorts its trajectory, several dozen or even hundreds of steps become necessary to preserve image quality, directly destroying FM's main advantage of few-step sampling.

### 2. The “likelihood-computation deadlock” caused by deterministic ODEs

- **The difficulty**: standard online RL, such as PPO/GRPO, must compute action log-likelihoods $\log\pi(a\mid s)$. Reverse denoising in DM is itself a stochastic Gaussian transition and naturally provides a computable probability density.
- **Mathematical deadlock**: FM generation is a purely deterministic ODE. Given state $x_t$, its probability of transitioning to the next state $x_{t-\Delta t}$ is mathematically a Dirac $\delta$ function. Taking the log of the $\delta$ function produces infinity (degenerate likelihood). Standard RL algorithms therefore cannot run on pure flow matching at all. The underlying architecture must be forcibly modified, for example by artificially injecting exploration noise through ODE-to-SDE conversion, making the originally elegant and simple FM approach unusually cumbersome and unstable.

Of course, a flow-matching model's fixed output for a given input does not mean that its output has no probability distribution. When we say that “the denoising process is fixed” in flow matching, we mean that, given a particular initial noise $z$, its trajectory toward the target data $x_1$ is unique and deterministic, because we solve an ODE rather than an SDE.

The model nonetheless has a distribution because its input source is a distribution. Mathematically, this is called a push-forward measure. Imagine standard Gaussian noise $z\sim\mathcal{N}(0,I)$ as a handful of sand scattered onto a conveyor belt with a particular wind direction, namely the vector field $v_t(x)$. Each grain's wind-driven trajectory is completely deterministic and fixed, but the probabilities of grains initially landing at different positions differ. When they reach the endpoint, they naturally accumulate into a new shape: the model-generated data distribution $p_1(x)$.

## III. FlowGRPO: Turning ODEs into SDEs

### (1) How to convert an ODE into an SDE

- Background principle: in standard flow matching or diffusion models, image generation is modeled as a deterministic ODE. Given initial pure noise, the trajectory along which the model denoises it into an image is uniquely determined and cannot change.
- The RL difficulty and its solution: reinforcement learning (RL) is fundamentally about exploration and exploitation. If the trajectory is deterministic, the model cannot try new generation paths and cannot discover which paths yield higher rewards. FlowGRPO therefore converts this deterministic ODE into an SDE. By introducing random noise perturbations controlled by $g(t)$ at each denoising step, it lets the model deviate from its established route and “learn by trial and error.” This stochastic exploration allows RL to work.

Specifically, FlowGRPO transforms the ODE in flow matching:

$$
dx_t=v_\theta(x_t,t)dt
$$

into the SDE:

$$
dx_t=\left[v_\theta(x_t,t)+g(t)\nabla\log p_t(x_t)\right]dt+\sqrt{2g(t)}\,dw_t
$$

Closer inspection shows that the added terms come from the stochastic gradient Langevin dynamics expression for random noise addition in diffusion:

$$
dx_t=\nabla\log p(x_t)dt+\sqrt{2}\,dw_t
$$

The second term is exploration noise, while the first is a correction that keeps the marginal distribution of $x$ with respect to $t$ unchanged, preventing out-of-distribution (OOD) situations.

The multiplicative coefficient $g(t)$ controls the intensity of Brownian motion and is a predefined function.

### (2) Ratio normalization and repairing the clipping mechanism

- Background principle: RL algorithms such as PPO and GRPO must use clipping to ensure stable training. By computing the importance ratio between the new and old policies, clipping limits the magnitude of each parameter update and prevents a chance high reward from destroying the model's original generation capability.

In SDE sampling for flow matching, each denoising transition is Gaussian. Let the current time interval be $\Delta t$ and the noise variance be $\sigma_{t_k}^2\Delta t$. The probability density functions of the new policy (the model currently being optimized) and the old policy (the reference model used for data collection) can both be written in Gaussian form. The importance ratio $r_{t_k}(\theta)$ is the ratio of the new policy's probability to the old policy's probability. Taking its natural logarithm gives:

$$
\begin{aligned}
\log r_{t_k}(\theta)
&=
-\frac{1}{2\sigma_{t_k}^2\Delta t}
\left\|x_{t_k-\Delta t}-\mu_\theta\right\|^2
\\
&\quad+
\frac{1}{2\sigma_{t_k}^2\Delta t}
\left\|x_{t_k-\Delta t}-\mu_{\theta_{old}}\right\|^2
\end{aligned}
$$

The key point is that the training trajectory $x_{t_k-\Delta t}$ in RL is sampled from the old policy $p_{\theta_{old}}$. It must therefore satisfy:

$$
\begin{aligned}
x_{t_k-\Delta t}
&=
\mu_{\theta_{old}}+\sqrt{\sigma_{t_k}^2\Delta t}\cdot\epsilon
\end{aligned}
$$

Here, $\epsilon\sim\mathcal{N}(0,I)$ is standard Gaussian noise. Substituting this $x_{t_k-\Delta t}$ into the logarithmic formula and defining the difference between policy means as $\Delta\mu=\mu_{\theta_{old}}-\mu_\theta$, expansion and simplification give:

$$
\begin{aligned}
\log r_{t_k}(\theta)
&=
-\frac{\|\Delta\mu\|^2}{2\sigma_{t_k}^2\Delta t}
\\
&\quad-
\frac{\epsilon^T\Delta\mu}{\sigma_{t_k}\sqrt{\Delta t}}
\end{aligned}
$$

Taking the expectation over Gaussian noise $\epsilon$ eliminates the second term because $\mathbb{E}[\epsilon]=0$:

$$
\begin{aligned}
\mathbb{E}_{\epsilon}\left[\log r_{t_k}(\theta)\right]
&=
-\frac{\|\Delta\mu\|^2}{2\sigma_{t_k}^2\Delta t}
\end{aligned}
$$

Since $\|\Delta\mu\|^2>0$ (the means differ whenever the policy is updated), the expected log importance ratio is strictly less than 0.

- The clipping-failure crisis: in flow-matching models, the properties of the probability density cause the importance-ratio distribution to shift systematically to the left (with mean below 1), with highly inconsistent variances across time-step sizes. The predefined clipping bounds (such as $[1-\epsilon,1+\epsilon]$) then become ineffective, unable to constrain overconfident, incorrect updates. This leads to severe reward hacking: the model exploits loopholes to obtain high scores while generating meaningless outputs.

- The RatioNorm solution: to address this, the paper introduces RatioNorm. It standardizes the log importance ratio and forces its distribution back toward a center near zero. The clipping bounds can then accurately constrain excessive policy updates again, ensuring stable convergence of the final image objective $\mathcal{J}_{Flow}(\theta)$.

$$
\begin{aligned}
\log \tilde{r}_{t_k}(\theta)
&=
\sigma_{t_k}\sqrt{\Delta t}
\left(
\log r_{t_k}(\theta)
\right.
\\
&\quad\left.
+\frac{\|\Delta\mu_\theta(x_{t_k},t_k)\|^2}{2\sigma_{t_k}^2\Delta t}
\right)
\end{aligned}
$$

## References

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR.
- Black, K., Janner, M., Du, Y., Kostrikov, I., & Levine, S. (2023). [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301). arXiv:2305.13301.
- Liu, J., Liu, G., Liang, J., et al. (2025). [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470). arXiv:2505.05470.
