---
title: "21.5 Frontier Research in Reinforcement Fine-Tuning"
chapter_title: "Frontiers of Reinforcement Fine-Tuning"
section_id: "21-05"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/21.强化微调的前沿探索/21.5 强化微调前沿研究.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 21.5 Frontier Research in Reinforcement Fine-Tuning

## I. Maximum Likelihood Reinforcement Learning (MaxRL)

### (I) The Core Idea

In RLVR, binary-reward tasks such as mathematics and coding provide only two rewards: correct (1) and incorrect (0). Yet the objective is to maximize the average reward (the probability of answering correctly), which appears mismatched. Drawing on cross-entropy loss in LLMs, we can also view RLVR as a classification task and introduce cross-entropy loss to bring the output distribution closer to the actual distribution. For a sample whose probability of predicting a correct answer is $p$, the cross-entropy loss is $-\log p$. This offers two advantages:

1. Larger gradients for tasks the model has not yet mastered

When $p$ is close to 0, the derivative of $\log(p)$ is larger. Samples with low probabilities of a correct answer receive larger "corrective" gradients, focusing training on tasks the model cannot yet solve.

2. Avoiding "mode collapse"

Traditional RL optimizes $p$ itself. If the model explores two nearly correct paths $z_1$ and $z_2$, RL only needs to increase the probability $p$ of the final sampled outcome. It can easily concentrate on one of $z_1$ or $z_2$, improve that path, and assign it a very high probability, while almost abandoning improvement and sampling of the other. This can also make $p$ large, but weakens robustness and generalization and may even cause "mode collapse."

MaxRL optimizes $\log(p)$. If a nearly correct path has a very low probability $p$ of producing the correct answer, $\log(p)$ provides a large gradient, forcing the model to actively improve rather than abandon that path. Ultimately, multiple different correct paths lead to the correct answer, avoiding "mode collapse" as much as possible.

### (II) Objective Function

**Define the objective**: suppose the model's probability of generating a correct answer for input $x$ is $p_\theta(x)$. Our objective is to maximize the log-likelihood of correct outcomes across all inputs $x$:

$$
J_{ML}(\theta)=\mathbb{E}_{x\sim D}[\log p_\theta(x)]
$$

**Derivation**:

1. **Use the linearity of expectation**: move the gradient operator $\nabla_\theta$ inside the expectation:

$$
\nabla_\theta J_{ML}(\theta)=\mathbb{E}_x[\nabla_\theta \log p_\theta(x)]
$$

2. **Apply the log-derivative trick from calculus**: by the chain rule, for any positive function $f(\theta)$, $\nabla_\theta \log f(\theta)=\frac{\nabla_\theta f(\theta)}{f(\theta)}$. Here, $f(\theta)=p_\theta(x)$, the model's probability of obtaining a correct outcome for input $x$.

3. **Substitute into the formula**:

$$
\nabla_\theta \log p_\theta(x)=\frac{1}{p_\theta(x)}\nabla_\theta p_\theta(x)
$$

4. **Final form**:

$$
\nabla_\theta J_{ML}=\mathbb{E}_x\left[\frac{1}{p_\theta(x)}\nabla_\theta p_\theta(x)\right]
$$

The practical meaning is that, to increase the log-probability of correctness, we should update parameters in the direction that increases $p_\theta(x)$, but multiply the update step size by $1/p_\theta(x)$. Samples with smaller probabilities receive larger step sizes (weights).

### (III) Advantage Function and Workflow

GRPO calculates an advantage for every sample in a group. We now derive the MaxRL advantage:

**Step 1: Calculate the gradient.** Differentiate with respect to $\theta$:

$$
\nabla_\theta J_{ML}=\mathbb{E}_x\left[\frac{1}{p_\theta(x)}\nabla_\theta p_\theta(x)\right]
$$

**Step 2: Expand $\nabla_\theta p_\theta(x)$.** In binary-feedback tasks, $p_\theta(x)$ is the probability of generating a correct answer, that is, the expected return $\mathbb{E}_{z\sim\pi_\theta}[R(z)]$. Using the classical log-derivative trick in reinforcement learning:

$$
\nabla_\theta p_\theta(x)=\nabla_\theta\int \pi_\theta(z|x)R(z)dz
=\mathbb{E}_{z\sim\pi_\theta}\left[R(z)\nabla_\theta\log \pi_\theta(z|x)\right]
$$

**Step 3: Substitute back and introduce a baseline.** Substitute the result of Step 2 into Step 1:

$$
\nabla_\theta J_{ML}=\mathbb{E}_x\left[\frac{1}{p_\theta(x)}\mathbb{E}_z\left[R(z)\nabla_\theta\log \pi_\theta(z|x)\right]\right]
$$

To reduce estimation error, we can introduce a baseline $b$ independent of the particular sample $z$. In RL, the most common baseline is the expected return itself, $b=p_\theta(x)$. Since $\mathbb{E}_z[\nabla_\theta\log\pi_\theta]=0$, subtracting the baseline does not change the expectation:

$$
\nabla_\theta J_{ML}=\mathbb{E}_x\left[\mathbb{E}_z\left[\frac{R(z)-p_\theta(x)}{p_\theta(x)}\nabla_\theta\log\pi_\theta(z|x)\right]\right]
$$

**Step 4: Sample estimation.** In actual training, we estimate this by sampling $N$ trajectories. The best unbiased estimate of $p_\theta(x)$ is the average return $\bar{r}=\frac{1}{N}\sum_j r_j$. Thus, the coefficient (advantage) of the gradient contribution from sample $j$ is:

$$
A_j^{MaxRL}=\frac{r_j-\bar{r}}{\bar{r}+\epsilon}
$$

Comparison:

GRPO:

$$
\mathrm{advantage}=\frac{\mathrm{reward}-\mathrm{mean\_reward}}{\mathrm{std\_reward}+\epsilon}
$$

MaxRL:

$$
\mathrm{advantage}=\frac{\mathrm{reward}-\mathrm{mean\_reward}}{\mathrm{mean\_reward}+\epsilon}
$$

## II. Experience-Based Reinforcement Learning (ExGRPO)

RLVR inherently wastes experience: reasoning trajectories generated by a model are discarded after a single round of gradient updates, never revisited later as humans revisit past work. Teaching the model to "learn something new by reviewing the old," using a "mistake notebook" to internalize every valuable successful experience, is crucial for training efficiency and capability improvement. The ExGRPO framework is introduced below.

### (I) Selecting Experience

Experience should be selected from problems of appropriate difficulty (model accuracy of 25%–75%). As in human learning, repeatedly consolidating already mastered content or starting with overly difficult material is ineffective.

For different reasoning paths on the same problem, research finds that average token entropy is an excellent metric: among all correct solutions, select the trajectory with the lowest entropy. This is because solutions with more logically correct reasoning have significantly lower entropy. High-entropy correct trajectories are often merely lucky guesses; repeatedly learning these trajectories does not help and may instead contaminate the model's logical abilities.

ExGRPO builds an "experience replay buffer," like a large "mistake notebook," collecting all successful reasoning cases during training. Using a Gaussian probability model, it preferentially samples problems from groups of moderate difficulty. Meanwhile, as the model improves, problems it has fully mastered (for example, answering all attempts correctly several times in succession) are removed from the learning queue, keeping its focus on more challenging tasks.

### (II) Objective Function

ExGRPO builds on GRPO. For an input query $q$, the current reference policy $\pi_{\theta_{\mathrm{old}}}$ generates a set of $K$ trajectories, $\mathcal{G}_q=\{o_i\}_{i=1}^{K}$.

Each trajectory's advantage $\hat{A}$ is calculated through within-group normalization (following Dr.GRPO, the paper removes length and standard-deviation normalization, retaining only mean centering):

$$
\hat{A}(o_i,\mathcal{G}_q)=r(q,o_i)-\mu_{\mathcal{G}_q}
$$

ExGRPO uses a mixed-policy optimization objective. In addition to correcting importance sampling, each training iteration allocates part of the minibatch's computation to exploring entirely new problems (on-policy) and another part to learning carefully selected experiences from the buffer (off-policy), balancing learning new material with reviewing old material.

At each parameter update, ExGRPO constructs a mixed minibatch $\mathcal{B}$ containing on-policy samples $\mathcal{B}_{\mathrm{on}}$ from the current dataset $\mathcal{D}$ and historical experience samples $\mathcal{B}_{\mathrm{exp}}$ from replay buffer $\mathcal{E}$. These are mixed according to hyperparameter $\rho\in[0,1)$.

$$
\begin{aligned}
\mathcal{J}_{\mathrm{ExGRPO}}(\theta)
=&(1-\rho)\cdot
\mathbb{E}_{q\sim\mathcal{B}_{\mathrm{on}}}
\left[
\frac{1}{K}\sum_{i=1}^{K}
\mathrm{CLIP}\left(w_i(\theta),\hat{A}(o_i,\mathcal{G}_q)\right)
\right] \\
&+\rho\cdot
\mathbb{E}_{q^{\ast}\sim\mathcal{B}_{\mathrm{exp}}}
\left[
\frac{1}{K}
\left(
\mathrm{CLIP}\left(w^{\ast}(\theta),\hat{A}(o^{\ast},\mathcal{G}_{q^{\ast}})\right)
+\sum_{i=1}^{K-1}
\mathrm{CLIP}\left(w_i(\theta),\hat{A}(o_i,\mathcal{G}_{q^{\ast}})\right)
\right)
\right]
\end{aligned}
$$

The first term is the on-policy objective:

- **Weight**: accounts for a proportion $1-\rho$ of the total objective.
- **Workflow**: for a current problem $q$ sampled from $\mathcal{B}_{\mathrm{on}}$, model $\pi_{\theta_{\mathrm{old}}}$ generates $K$ trajectories in real time and calculates the GRPO loss.
- **Importance sampling**: $w_i(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,\lt t})}$ limits the magnitude of policy updates.

The second term is the experiential off-policy objective:

- **Weight**: accounts for a proportion $\rho$ of the total objective.
- **Mixed-group construction**: for a historical problem $q^{\ast}$ sampled from the buffer, ExGRPO constructs a mixed advantage-estimation group $\mathcal{G}_{q^{\ast}}=\{o^{\ast}\}\cup\{o_i\}_{i=1}^{K-1}$. Here, $o^{\ast}$ is a high-quality historical trajectory generated by a past policy $\pi_{\theta_{\mathrm{past}}}$, while the remaining $K-1$ trajectories $o_i$ are generated in real time by the current reference policy $\pi_{\theta_{\mathrm{old}}}$.
- **Off-policy importance correction**: to eliminate distribution shift from directly using historical data and ensure unbiased gradients, the importance weight for replayed trajectory $o^{\ast}$ is specifically reweighted as the ratio between the current policy and the past policy that generated it: $w^{\ast}(\theta)=\frac{\pi_\theta(o_t^{\ast}\mid q^{\ast},o_{\lt t}^{\ast})}{\pi_{\theta_{\mathrm{past}}}(o_t^{\ast}\mid q^{\ast},o_{\lt t}^{\ast})}$.

### (III) Policy Shaping

To address the possibility that directly optimizing low-entropy historical trajectories harms RL exploration, the second term of the objective above is changed to:

$$
\rho\cdot
\mathbb{E}_{q^{\ast}\sim\mathcal{B}_{\mathrm{exp}}}
\left[
\frac{1}{K}
\left(
f\left(w^{\ast}(\theta)\right)\cdot\hat{A}(o^{\ast},\mathcal{G}_{q^{\ast}})
+\sum_{i=1}^{K-1}
\mathrm{CLIP}\left(w_i(\theta),\hat{A}(o_i,\mathcal{G}_{q^{\ast}})\right)
\right)
\right]
$$

In the latter part of the formula (the off-policy experience objective), the original $\mathrm{CLIP}\left(w^{\ast}(\theta),\hat{A}(o^{\ast},\mathcal{G}_{q^{\ast}})\right)$ is replaced by $f\left(w^{\ast}(\theta)\right)\cdot\hat{A}(o^{\ast},\mathcal{G}_{q^{\ast}})$.

Here, $f(x)$ is a nonlinear transformation introducing a small constant $\beta$ (usually set to 0.1 in practice):

$$
f(x)=\frac{x}{x+\beta}
$$

Replacing the standard PPO/GRPO $\mathrm{CLIP}$ mechanism with $f(w)$ is motivated by the following key mathematical properties and their RL benefits:

- **Suppressing high-probability signals (preventing overfitting)**: when importance weight $w$ is large (the current policy is already strongly inclined to generate this historical action), $f(w)$ approaches 1. It therefore acts similarly to clipping, suppressing extreme weights, controlling variance from off-policy learning, and preventing excessive updates on an already highly certain trajectory.
- **Amplifying low-probability signals (encouraging novel exploration)**: when $w$ is small and near 0 (the current policy is unlikely to generate the action, but it was a good action in historical experience), the presence of $\beta$ (such as 0.1) gives $f(w)\approx\frac{w}{\beta}$, locally amplifying these weak signals.
- **Dynamic tradeoff**: this design lets the model learn from parts of experience trajectories that remain "novel" to the current policy, rather than blindly reinforcing already high-probability actions, preserving policy entropy (exploration) while exploiting experience.

## III. Reinforcement Learning That Accumulates Experience and Reflection through Self-Distillation

### (I) Background

Reinforcement learning (RL) has become a core method for language models to learn from environmental rewards or feedback. However, for agent reasoning and decision-making tasks, especially early in a task, standard RL with verifiable rewards (RLVR) faces significant challenges: in practice, environmental feedback is often a sparse, delayed scalar signal with extremely little information. This can trap the model in a blind, repetitive trial-and-error cycle, making lasting behavioral corrections difficult.

We can therefore embed an explicit "experience–reflection–consolidation" cycle into reinforcement learning, especially early on. Instead of relying only on the final scalar outcome, environmental feedback is converted into structured intermediate reasoning signals (reflections), guiding local behavioral corrections within the same episode. Successful corrections are ultimately internalized into the base policy, effectively turning thin scalar feedback into concrete behavioral correction strategies.

### (II) Algorithm Workflow

#### 1. First Attempt

For a given task input $x$, language model $\pi_\theta$ first generates an initial attempt $y^{(1)}$:

$$
y^{(1)}\sim\pi_\theta(\cdot\mid x)
$$

The model then receives textual feedback $f^{(1)}$ and reward $r^{(1)}$ from the environment. It optimizes the first attempt with a standard RL objective. The RL loss is defined as:

$$
\mathcal{L}_{\mathrm{policy}}(\theta)=-\mathbb{E}\left[A\log\pi_\theta(y\mid x,\cdot)\right]
$$

Here, $A$ is an advantage estimate calculated from the reward.

#### 2. Gated Reflection

ERL does not reflect on every attempt. Instead, it sets a threshold $\tau$ (for example, $\tau=1$). Reflection is triggered only when the first attempt's reward $r^{(1)}\lt\tau$ (poor performance or failure). This avoids reward hacking on already successful trajectories and stabilizes training.

Once reflection is triggered, the model generates structured self-reflection $\Delta$:

$$
\Delta\sim\pi_\theta(\cdot\mid x,y^{(1)},f^{(1)},r^{(1)},m)
$$

A cross-episode reflection memory $m$ is introduced here. If a reflection leads to a successful second attempt (a reward exceeding threshold $\tau$), it is retained and stored in memory. Future episodes can retrieve and reuse these validated correction strategies as contextual priors, stabilizing reflection generation and accumulating knowledge throughout training.

#### 3. Second Attempt

Based on the generated reflection $\Delta$, the model makes a more precise second attempt $y^{(2)}$:

$$
y^{(2)}\sim\pi_\theta(\cdot\mid x,\Delta)
$$

This attempt receives new feedback and reward $(f^{(2)},r^{(2)})$. The reflection itself is assigned reward $\bar{r}=r^{(2)}$, encouraging reflections that improve downstream performance. If the second attempt succeeds ($r^{(2)}\gt\tau$), reflection $\Delta$ is stored in memory $m$, accumulating knowledge. The second attempt and reflection process are also updated through RL using the same $\mathcal{L}_{\mathrm{policy}}(\theta)$ above.

#### 4. Internalization and Consolidation

During actual deployment (inference), the model cannot obtain environmental feedback or explicit reflection prompts. To let it "remember" correction strategies discovered during training, ERL introduces an internalization mechanism.

Internalization is implemented through selective knowledge distillation. The model is supervised to imitate only successful second attempts, but with reflection context removed from the input. The distillation loss is:

$$
\mathcal{L}_{\mathrm{distill}}(\theta)=-\mathbb{E}\left[I(r^{(2)}\gt 0)\log\pi_\theta(y^{(2)}\mid x)\right]
$$

Here, $I(\cdot)$ is an indicator function. This forces the base policy $\pi_\theta$ to learn to output the improved behavior $y^{(2)}$ directly from the original input $x$, ensuring that behavioral improvements persist even without reflection at test time.

Throughout the trajectory, the first attempt, the reflection itself, and the second attempt are all optimized through the RL objective, embedding a complete "experience–reflection–consolidation" cycle within a single RL trajectory. After receiving failure feedback from the environment, the model must first generate structured self-reflection to guide a refined second attempt. This means that it learns not only "good outcomes" but also continually improves its ability to "generate effective reflections" using environmental rewards.

## References

- Tajwar, F., Zeng, G., Zhou, Y., et al. (2026). [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710). arXiv:2602.02710.
- Zhan, R., Li, Y., Wang, Z., Qu, X., Liu, D., Shao, J., Wong, D. F., & Cheng, Y. (2025). [ExGRPO: Learning to Reason from Experience](https://arxiv.org/abs/2510.02245). arXiv:2510.02245.
- Shi, T., Chen, S., Jiang, B., et al. (2026). [Experiential Reinforcement Learning](https://arxiv.org/abs/2602.13949). arXiv:2602.13949.
