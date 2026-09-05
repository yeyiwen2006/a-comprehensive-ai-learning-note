---
title: "26.7 Reinforcement Learning in Diffusion Models and DPO for Diffusion"
chapter_title: "Diffusion Models"
section_id: "26-07"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/26.扩散模型/26.7 扩散模型中的强化学习与DPO for Diffusion.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 26.7 Reinforcement Learning in Diffusion Models and DPO for Diffusion

## I. Difficulties of RL in Diffusion Models

### 1. Credit assignment over long trajectories with sparse rewards

- **The difficulty**: RL reward signals, such as aesthetic scores and human-preference alignment scores $R(x_0)$, are extremely sparse and can only be obtained once generation has finished and the final complete image is available. With no immediate, dense intermediate rewards, a low score for the final image makes it very difficult for the RL algorithm to identify the step at which the neural network output went wrong: for example, the coarse-outline stage at $t=0.9$ or the detailed-texture stage at $t=0.1$.
- **Consequence**: conventional advantage estimators, such as GAE, struggle to propagate reward gradients accurately backward through such long chains of ODE/SDE solver steps. Optimization becomes extremely inefficient, and credit may even be assigned incorrectly.

Value-network training is especially prone to collapse, which is why GRPO-based algorithms are more commonly used.

### 2. The exploration crisis in extremely high-dimensional continuous action spaces

- **The difficulty**: effective random exploration in such a vast, high-dimensional continuous space is like finding a needle in a haystack. Blindly adding Gaussian exploration noise to the vector field or noise predicted at each step can easily introduce cascading errors into the entire differential-equation integration trajectory, causing the final image to collapse completely into meaningless noise.
- **Consequence**: to prevent the generation trajectory from collapsing, researchers must set the exploration-noise variance extremely low. This, however, deprives the model of exploration capability and quickly traps it in local optima. It is also the underlying reason why DM+RL is highly prone to mode collapse and loss of generation diversity.

## II. DPO for Diffusion

### (1) Mathematical formulation

Step 1: Reparameterize the RLHF objective

In a standard RL framework with a KL penalty, we want to fine-tune a policy (diffusion model) $p_\theta(x\mid c)$ so that, given a conditioning prompt $c$, its generated image $x$ maximizes the implicit reward $r(x,c)$ while not deviating too far from the pretrained reference model $p_{\mathrm{ref}}(x\mid c)$:

$$
\begin{aligned}
\max_\theta\ \mathbb{E}_{x\sim p_\theta}\left[r(x,c)\right]
&\quad-\beta D_{\mathrm{KL}}\left(p_\theta(x\mid c)\Vert p_{\mathrm{ref}}(x\mid c)\right)
\end{aligned}
$$

Here, $\beta$ is a hyperparameter controlling the degree of deviation. This optimization problem has a theoretical closed-form optimal policy:

$$
p^{*}(x\mid c)=\frac{1}{Z}p_{\mathrm{ref}}(x\mid c)\exp\left(\frac{1}{\beta}r(x,c)\right)
$$

Here, $Z$ is the partition function. The key step in DPO is to rearrange this expression and solve for the reward function $r(x,c)$:

$$
r(x,c)=\beta\log\frac{p^{*}(x\mid c)}{p_{\mathrm{ref}}(x\mid c)}+\beta\log Z
$$

Step 2: Substitute into the Bradley–Terry preference model

In preference learning, we usually assume that human preferences follow the Bradley–Terry (BT) model. For a given prompt $c$, the probability that a human considers image $x_w$ (the winner) better than $x_l$ (the loser) is:

$$
p(x_w\succ x_l\mid c)=\sigma\left(r(x_w,c)-r(x_l,c)\right)
$$

Here, $\sigma$ is the sigmoid function. Substituting the expression for $r(x,c)$ from Step 1 into the BT model cancels the constant term $\beta\log Z$ because a difference is being computed:

$$
\begin{aligned}
p(x_w\succ x_l\mid c)
&=\sigma\left(
\beta\log\frac{p_\theta(x_w\mid c)}{p_{\mathrm{ref}}(x_w\mid c)}
-\beta\log\frac{p_\theta(x_l\mid c)}{p_{\mathrm{ref}}(x_l\mid c)}
\right)
\end{aligned}
$$

This is the standard DPO objective used in large language models. Diffusion models, however, face a key problem: their exact log-likelihood $\log p_\theta(x\mid c)$ is mathematically difficult to compute.

Step 3: Substitute the diffusion model's ELBO

The main contribution of Diffusion-DPO is to use the evidence lower bound (ELBO) to approximate this intractable log-likelihood. In diffusion models, maximizing log-likelihood is equivalent to minimizing denoising prediction error (the MSE loss). Let $\mathcal{L}_\theta(x,c,t)$ be the model's denoising loss at time step $t$ (using noise prediction $\epsilon$ as an example):

$$
\begin{aligned}
\mathcal{L}_\theta(x,c,t)
&=\left\lVert \epsilon_\theta(x_t,c,t)-\epsilon\right\rVert_2^2
\end{aligned}
$$

Because likelihood is negatively related to denoising loss, we can make the following approximation:

$$
\begin{aligned}
\log p_\theta(x\mid c)
&\approx -\mathbb{E}_{t,\epsilon}\left[\mathcal{L}_\theta(x,c,t)\right]+C
\end{aligned}
$$

The log-likelihood ratio can therefore be replaced by a difference in denoising losses:

$$
\begin{aligned}
\log\frac{p_\theta(x\mid c)}{p_{\mathrm{ref}}(x\mid c)}
&\approx
\mathbb{E}_{t,\epsilon}
\left[
\mathcal{L}_{\mathrm{ref}}(x,c,t)-\mathcal{L}_\theta(x,c,t)
\right]
\end{aligned}
$$

The physical intuition is clear: if the loss difference between the reference and current models on the “good image” $x_w$ is greater than their loss difference on the “bad image” $x_l$, we apply a reward; otherwise, we apply a penalty.

The same applies to the flow-matching models discussed later:

In flow matching, exact log-likelihood remains difficult to compute, but the MSE loss for fitting the vector field can replace the diffusion model's noise-prediction loss. We simply replace $\mathcal{L}_\theta$ with the flow-matching objective discussed in DMD:

$$
\begin{aligned}
\mathcal{L}^{\mathrm{FM}}_\theta(x,c,t)
&=\left\lVert v_\theta(x_t,c,t)-(x_1-x_0)\right\rVert_2^2
\end{aligned}
$$

Substituting this vector-field error into the Diffusion-DPO framework (called Flow-DPO in recent literature) allows offline preference pairs to fine-tune flow-matching models directly, without the complex process of solving ODEs through online RL.

### (2) Workflow

Stage 1: Data preparation

1. Build or collect a static offline preference dataset $\mathcal{D}$. It contains many tuples $(c,x_w,x_l)$: a “text prompt,” a “good image preferred by humans or AI,” and a “bad image rejected by humans or AI.”
2. Prepare a pretrained base diffusion model as the reference model $\theta_{\mathrm{ref}}$ and freeze its weights.
3. Initialize a policy model $\theta$ with exactly the same architecture, usually from $\theta_{\mathrm{ref}}$. This is the model to be trained.

Stage 2: Training loop (batch level)

At each training step, perform the following:

1. Sampling and noise addition: sample a batch of $(c,x_w,x_l)$ from the dataset. Randomly sample a time step $t\sim\mathcal{U}(0,T)$ and Gaussian noise $\epsilon\sim\mathcal{N}(0,I)$.
2. Construct intermediate states: use the forward-diffusion formulas to add noise to both the good and bad images up to time step $t$:

$$
\begin{aligned}
x_{w,t} &= \sqrt{\bar{\alpha}_t}x_w+\sqrt{1-\bar{\alpha}_t}\epsilon,\\
x_{l,t} &= \sqrt{\bar{\alpha}_t}x_l+\sqrt{1-\bar{\alpha}_t}\epsilon.
\end{aligned}
$$

3. Compute reference-model errors (no gradient): feed the noisy good and bad images, together with $t$ and $c$, into the frozen reference model $\theta_{\mathrm{ref}}$. Compute the mean squared errors between its noise predictions and the actual noise $\epsilon$: $\mathcal{L}_{\mathrm{ref}}(x_w)$ and $\mathcal{L}_{\mathrm{ref}}(x_l)$.
4. Compute policy-model errors (with gradients): feed the same data into the policy model $\theta$ being trained, and compute its prediction errors: $\mathcal{L}_\theta(x_w)$ and $\mathcal{L}_\theta(x_l)$.
5. Update through backpropagation: substitute these four scalar errors into the $\mathcal{L}_{\mathrm{DPO}\text{-}\mathrm{Diff}}(\theta)$ formula. Use the automatic differentiation mechanism of a deep learning framework (such as PyTorch) to compute gradients with respect to $\theta$, and update the model weights with an optimizer (such as AdamW).

## References

- Black, K., Janner, M., Du, Y., Kostrikov, I., & Levine, S. (2023). [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301). arXiv:2305.13301.
- Wallace, B., Dang, M., Rafailov, R., et al. (2024). [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908). CVPR 2024.
- Liu, J., Liu, G., Liang, J., et al. (2025). [Improving Video Generation with Human Feedback](https://arxiv.org/abs/2501.13918). NeurIPS.
