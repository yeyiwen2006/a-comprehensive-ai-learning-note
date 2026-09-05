---
title: "13.4 Other Applications of KL Divergence in RL"
chapter_title: "Combining Value and Policy Methods"
section_id: "13-04"
language: en
source_language: zh
source_docx: "第2部分 强化学习/13.综合价值与策略的算法/13.4 KL散度在RL中的其他应用.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 13.4 Other Applications of KL Divergence in RL

Beyond stability constraints in TRPO and PPO, KL divergence has several other ingenious applications in RL.

## 1. Exploration and Information Gain

In information-gain-based exploration, KL divergence measures how much an observation changes the agent's beliefs about environmental dynamics parameters. VIME is a representative method.

* Core idea: explore states providing the most “new information.”
* Implementation: maintain a posterior over dynamics parameters $\theta$ and compare parameter posteriors before and after observing a new transition.
* If transition $(s_t,a_t,s_{t+1})$ substantially changes beliefs about dynamics parameters, it provides high information gain and earns an extra intrinsic reward.

Mathematically:

$$
r_{\mathrm{intrinsic},t}
=
D_{\mathrm{KL}}\bigl(
p(\theta\mid \xi_t,a_t,s_{t+1})
\Vert
p(\theta\mid \xi_t)
\bigr)
$$

Here $\xi_t$ is interaction history through time $t$, and $\theta$ denotes dynamics-model parameters. The expression compares parameter posteriors before and after adding a transition, rather than directly taking KL between a continuous predictive distribution and a Dirac distribution representing one observation.

This encourages actively exploring environmental regions whose outcomes cannot be predicted accurately, learning the world model more efficiently.

## 2. Imitation Learning and Behavior Cloning

Imitation learning aims to make agent policy $\pi_{\mathrm{learner}}$ as close as possible to expert policy $\pi_{\mathrm{expert}}$.

* Direct KL minimization: define the loss as their KL divergence:

$$
L_{\mathrm{imitation}}
=
\mathbb{E}_{s\sim d_{\mathrm{expert}}}
\bigl[
D_{\mathrm{KL}}\bigl(
\pi_{\mathrm{expert}}(\cdot\mid s)
\Vert
\pi_{\mathrm{learner}}(\cdot\mid s)
\bigr)
\bigr]
$$

* Interpretation: the learner's action distribution must closely match the expert's on expert-visited states. This is a direct, effective form of imitation learning.

## 3. Information-Theoretic Regularization and Maximum-Entropy RL

Maximum-entropy RL (such as SAC and Soft Q-Learning) maximizes not only cumulative reward but also policy entropy, or randomness.

* Core objective:

$$
J(\pi)
=
\mathbb{E}
\bigl[
\sum_t \gamma^{t}
\bigl(
r(s_t,a_t)+\alpha\mathcal{H}(\pi(\cdot\mid s_t))
\bigr)
\bigr]
$$

Here $\mathcal{H}$ is entropy and $\alpha$ the temperature parameter.

* Connection to KL: this maximum-entropy objective is mathematically equivalent to adding a KL regularizer relative to a uniform distribution to the standard RL objective.
* Maximizing $\mathcal{H}(\pi)$ is equivalent to minimizing $D_{\mathrm{KL}}(\pi\Vert\mathrm{Uniform})$, encouraging a policy close to uniform, the most random distribution.
* In SAC: the policy-update objective includes KL minimization relative to a guiding distribution, usually the Boltzmann distribution induced by the current Q function.

## 4. Multitask Learning and Knowledge Transfer

When an agent learns related tasks, policy distillation transfers behavioral knowledge from a teacher to a student. KL directly constrains the student's action distribution to match the teacher:

$$
L_{\mathrm{distill}}
=
\mathbb{E}_{s\sim\mathcal{D}}
\left[
D_{\mathrm{KL}}\bigl(
\pi_{\mathrm{teacher}}(\cdot\mid s)
\Vert
\pi_{\mathrm{student}}(\cdot\mid s)
\bigr)
\right]
$$

* Knowledge transfer: minimizing this loss reproduces the teacher's action distribution on states covered by $\mathcal{D}$.
* Preventing forgetting: an old-task teacher policy can help preserve old behavior, but effectiveness depends on state-data coverage of the old task.

Elastic Weight Consolidation (EWC) also mitigates catastrophic forgetting, but does not directly constrain KL between old and new policies. Instead, old-task Fisher information imposes a quadratic penalty on changes to important parameters:

$$
L_{\mathrm{EWC}}
=
L_{\mathrm{new\_task}}
+
\frac{\lambda}{2}
\sum_i F_i(\theta_i-\theta_i^{*})^{2}
$$

Here $\theta_i^{*}$ is the parameter at the end of old-task training, and $F_i$ measures parameter $i$'s importance to that task.

## References

- Houthooft, R., Chen, X., Duan, Y., Schulman, J., De Turck, F., & Abbeel, P. (2016). [VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674). NeurIPS 2016.
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html). AISTATS 2011.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). ICML 2018.
- Rusu, A. A., Colmenarejo, S. G., Gülçehre, Ç., et al. (2016). [Policy Distillation](https://arxiv.org/abs/1511.06295). ICLR 2016.
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). [Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114). Proceedings of the National Academy of Sciences, 114(13), 3521-3526.
