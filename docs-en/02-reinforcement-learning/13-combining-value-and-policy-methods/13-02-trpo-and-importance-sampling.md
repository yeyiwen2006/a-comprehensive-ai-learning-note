---
title: "13.2 TRPO and Importance Sampling"
chapter_title: "Combining Value and Policy Methods"
section_id: "13-02"
language: en
source_language: zh
source_docx: "第2部分 强化学习/13.综合价值与策略的算法/13.2 TRPO算法与重要性采样.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 13.2 TRPO and Importance Sampling

## I. Background

Reinforcement learning, especially policy gradient methods, improves agents by continually updating the policy network (a probability distribution). A major risk is that an excessively large update can send the new policy completely “off course,” choosing poor actions, sampling poor data, and producing inaccurate gradients. This vicious cycle can collapse training, sharply reduce performance, and make recovery difficult.

## II. Mathematical Formulation

In machine learning, KL divergence $D_{\mathrm{KL}}(P\Vert Q)=H(P,Q)-H(P)$ expresses the difference between prediction distribution $Q$ and label distribution $P$. Here $H(P,Q)$ is cross-entropy, the average length needed to encode data from $P$ using $Q$'s coding system. More precisely, this is forward KL divergence: the extra information required by using the wrong coding system ($Q$) rather than the correct one ($P$), measuring $Q$'s deviation from $P$. It strongly penalizes regions where the old policy $P$ has high probability and new policy $Q$ low probability, ensuring the new policy covers the old one without excessive collapse and avoiding omission of modes of the true distribution. Reverse KL divergence, $D_{\mathrm{KL}}(Q\Vert P)$, instead strongly penalizes regions where old $P$ has low probability and new $Q$ high probability.

In RL, KL divergence acts as a “stabilizer” or “brake.” Large models generally use reverse KL, penalizing high-new/low-old probability regions so that the new probability space remains within high-probability regions after pretraining. Leaving those regions often produces garbled or semantically abnormal outputs, namely “mode collapse.” The space obtained through RL is often a subset of pretraining's space, perhaps one reason RL struggles to teach skills completely absent from pretraining.

Given $D_{\mathrm{KL}}(P\Vert Q)=H(P,Q)-H(P)$, where $H(P,Q)$ is cross-entropy, why do TRPO and PPO use KL rather than cross-entropy? Not because their gradients differ (they are actually the same), but because their “scales” have different zero points. KL has an absolute zero ($D_{\mathrm{KL}}=0$ means no change), enabling a common safety threshold, the trust region. Cross-entropy's “zero point” floats with the old policy's entropy $H(P)$, preventing a fixed threshold on update size.

TRPO is the optimization problem:

$$
\begin{aligned}
\max_\theta\quad
& \mathbb{E}_{s\sim\rho_{\theta_{\mathrm{old}}},\,a\sim\pi_{\theta_{\mathrm{old}}}}
\bigl[
\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}
A^{\pi_{\mathrm{old}}}(s,a)
\bigr] \\
\text{subject to}\quad
& \bar{D}_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}\Vert \pi_\theta)\le \delta
\end{aligned}
$$

Here $\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}$ is the probability ratio, $A^{\pi_{\mathrm{old}}}(s,a)$ the advantage, and $\delta$ the trust-region radius, the maximum permitted policy change.

## III. Explaining the Objective: Importance Sampling

TRPO is an online policy method. As policy updates, old-policy samples become unusable, wasting substantial data. Importance sampling from statistics enables effective reuse.

Suppose you need to estimate the average income of the entire country.

* Target distribution $P$: the national population.
* Proposal distribution $Q$: as the interviewer, you can question random passersby only in Shanghai's Lujiazui district.

Problem with a direct average: Lujiazui passersby's average income will be far above the national average because the sampling location (distribution $Q$) favors wealthy people.

Importance sampling's solution: continue sampling in Lujiazui, but weight the accounting:

1. Wealthy people: common in Lujiazui but less prevalent nationally, so multiply their incomes by weights below $1$ (downweight).
2. Ordinary earners: rare in Lujiazui but common nationally. Finding one yields a valuable sample representing many people nationwide, so multiply the income by a weight above $1$ (upweight).

Suppose we need the expectation of $f(x)$ under target $P(x)$ but can sample only from $Q(x)$:

$$
\mathbb{E}_{x\sim P}[f(x)] = \int P(x)f(x)\,dx
$$

An identity transformation gives:

$$
\begin{aligned}
\mathbb{E}_{x\sim P}[f(x)]
&= \int P(x)f(x)\cdot \frac{Q(x)}{Q(x)}\,dx \\
&= \int Q(x)(\frac{P(x)}{Q(x)}f(x))\,dx
\end{aligned}
$$

Therefore:

$$
\mathbb{E}_{x\sim P}[f(x)]
=
\mathbb{E}_{x\sim Q}
\bigl[
\frac{P(x)}{Q(x)}f(x)
\bigr]
$$

$\frac{P(x)}{Q(x)}$ is the importance weight.

We can thus “reuse” data sampled under old policy Q.

Applied to TRPO:

$$
\mathbb{E}_{a\sim\pi_{\mathrm{old}}}
\bigl[
\frac{\pi_{\mathrm{new}}(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}
A(s,a)
\bigr]
$$

This expectation conditions on s and does not yet account for changes in s's occupancy measure as the policy changes. Including those gives:

$$
J(\theta)=
\sum_s\rho_{\pi_\theta}(s)
\sum_a\pi_\theta(a\mid s)r(s,a)
$$

After importance sampling:

$$
J(\theta)=
\mathbb{E}_{s\sim\rho_{\theta_{\mathrm{old}}},\,a\sim\pi_{\theta_{\mathrm{old}}}}
\bigl[
\frac{\rho_{\pi_\theta}(s)}{\rho_{\pi_{\theta_{\mathrm{old}}}}(s)}
\cdot
\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}
\cdot
A^{\pi_{\mathrm{old}}}(s,a)
\bigr]
$$

The difficulty is computing $\frac{\rho_{\pi_\theta}(s)}{\rho_{\pi_{\theta_{\mathrm{old}}}}(s)}$. State occupancy $\rho(s)$ is jointly determined by environmental dynamics $P(s'\mid s,a)$ and policy $\pi$ through multistep interaction, unlike action probabilities $\pi(a\mid s)$, which can be read directly from the policy network. Explicitly obtaining the state-distribution ratio is difficult without a complete environment model and solutions to complex flow equations.

TRPO directly approximates:

$$
\frac{\rho_{\pi_\theta}(s)}{\rho_{\pi_{\theta_{\mathrm{old}}}}(s)}\approx 1
$$

In other words, it assumes that the encountered state distribution has not changed dramatically at the instant the policy updates.

Importance sampling thus requires limiting Q's deviation from P. Otherwise, finding an extremely rare poor household in Lujiazui (Q) could give that sample a weight tens of thousands of times larger. One sample would dominate the gradient, making network updates fluctuate violently and collapsing training.

## IV. Solving the Optimization Problem

TPRO can be written as:

$$
\begin{aligned}
\max_\theta\quad
& L(\theta)=\mathbb{E}
\bigl[
\frac{\pi_\theta}{\pi_{\mathrm{old}}}A
\bigr] \\
\text{subject to}\quad
& \bar{D}_{\mathrm{KL}}(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)
\Vert
\pi_\theta(\cdot\mid s)
)\le \delta
\end{aligned}
$$

The solution is:

Step 1: Taylor expansion (approximation)

Let:

$$
x=\theta-\theta_{\mathrm{old}}
$$

The small trust region permits Taylor expansions of the objective and constraint.

1. First-order expansion of $L(\theta)$:

$$
L(\theta)\approx
L(\theta_{\mathrm{old}})
+
\nabla_\theta L(\theta_{\mathrm{old}})^{T}x
$$

Here $L(\theta_{\mathrm{old}})$ is constant and irrelevant to optimization; $\nabla_\theta L(\theta_{\mathrm{old}})$ is the standard policy gradient, denoted vector $\mathbf{g}$. The approximate objective is:

$$
\max_x\ \mathbf{g}^{T}x
$$

2. Second-order expansion of constraint $\bar{D}_{\mathrm{KL}}$:

$$
\bar{D}_{\mathrm{KL}}\bigl(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)
\Vert
\pi_{\theta_{\mathrm{old}}+x}(\cdot\mid s)
\bigr)
\approx
\bar{D}_{\mathrm{KL}}\bigl(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)
\Vert
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)
\bigr)
+
(\nabla_\theta \bar{D}_{\mathrm{KL}})^{T} x
+
\frac{1}{2}x^{T}\nabla_\theta^{2}\bar{D}_{\mathrm{KL}}x
$$

The first term is $\bar{D}_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)\Vert\pi_{\theta_{\mathrm{old}}}(\cdot\mid s))=0$. The second is at a minimum when old and new distributions coincide, so its derivative is $0$. In the third, $\nabla_\theta^{2}\bar{D}_{\mathrm{KL}}$ is the KL second-derivative matrix, the Fisher information matrix (FIM), denoted $\mathbf{H}$. The approximate constraint is:

$$
\frac{1}{2}x^{T}\mathbf{H}x\le \delta
$$

Step 2: finding the new direction (Lagrange multipliers)

The problem becomes:

$$
\begin{aligned}
\max_x\quad
& \mathbf{g}^{T}x \\
\text{subject to}\quad
& \frac{1}{2}x^{T}\mathbf{H}x\le \delta
\end{aligned}
$$

Construct the Lagrangian:

$$
\mathcal{L}(x,\lambda)=
\mathbf{g}^{T}x
-\lambda
\bigl(
\frac{1}{2}x^{T}\mathbf{H}x-\delta
\bigr)
$$

Differentiate with respect to $x$ and set the derivative to $0$:

$$
\mathbf{g}-\lambda\mathbf{H}x=0
\quad\Longrightarrow\quad
\mathbf{H}x=\frac{1}{\lambda}\mathbf{g}
$$

This gives the update direction, without its step size:

$$
x\propto\mathbf{H}^{-1}\mathbf{g}
$$

This is the famous natural gradient direction. Instead of following ordinary gradient $\mathbf{g}$, the inverse of the curvature matrix $\mathbf{H}$ corrects the direction.

Next determine the step size (scaling coefficient). Substitute:

$$
x=\beta\mathbf{H}^{-1}\mathbf{g}
$$

into $\frac{1}{2}x^{T}\mathbf{H}x=\delta$ to obtain:

$$
\beta=
\sqrt{
\frac{2\delta}{\mathbf{g}^{T}\mathbf{H}^{-1}\mathbf{g}}
}
$$

The final parameter update is:

$$
x^{*}
=
\sqrt{
\frac{2\delta}{\mathbf{g}^{T}\mathbf{H}^{-1}\mathbf{g}}
}
\mathbf{H}^{-1}\mathbf{g}
$$

Step 3: numerical solution (conjugate gradients, CG)

Although derived, the formula cannot be computed directly in practice. Hessian $\mathbf{H}$ is $N\times N$, where $N$ is the parameter count. With one million parameters, $\mathbf{H}$ has one trillion elements, exhausting memory even before computing $\mathbf{H}^{-1}$.

TRPO's key technique is conjugate gradients (CG). We need vector:

$$
\mathbf{s}=\mathbf{H}^{-1}\mathbf{g}
$$

Equivalently, solve:

$$
\mathbf{H}\mathbf{s}=\mathbf{g}
$$

CG solves $\mathbf{H}\mathbf{s}=\mathbf{g}$ without explicitly constructing $\mathbf{H}$; it needs only products $\mathbf{H}v$. A function taking $v$ and returning $\mathbf{H}v$ is sufficient to solve for $\mathbf{s}$ iteratively.

$\mathbf{H}$ is KL's second derivative. Computing $\mathbf{H}v$ does not require actually computing second derivatives; an efficient technique, the Pearlmutter trick, gives:

$$
\mathbf{H}v
=
\nabla_\theta(\nabla_\theta D_{\mathrm{KL}}\cdot v)
$$

Specifically:

1. Compute KL gradient $\nabla_\theta D_{\mathrm{KL}}$.
2. Take its dot product with $v$, giving a scalar.
3. Differentiate that scalar again.

Automatic differentiation frameworks such as PyTorch and TensorFlow need only two backpropagations, making this fast without storing the matrix.

Step 4: line search

The second-order approximation is only an approximation. After an actual update, KL may exceed $\delta$ or the true objective may decrease. For safety, TRPO finally performs backtracking line search:

1. Calculate ideal update $x^{*}$.
2. Set the actual update:

$$
\Delta\theta=\alpha^{j} x^{*}
\quad
(\alpha \text{ initially equals } 1)
$$

3. Check two conditions:

$$
\bar{D}_{\mathrm{KL}}(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s)
\Vert
\pi_{\theta_{\mathrm{old}}+\Delta\theta}(\cdot\mid s)
)\le \delta
$$

$$
L(\theta_{\mathrm{old}}+\Delta\theta)\ge L(\theta_{\mathrm{old}})
$$

4. If they fail, set $j\leftarrow j+1$, for example changing $\alpha$ to $0.5,0.25,\ldots$, shortening the step and retrying.
5. Update parameters only after the conditions hold.

Summary of TRPO's solution procedure:

1. Prepare data: collect trajectories under the old policy.
2. Compute gradient: obtain policy gradient $\mathbf{g}$.
3. Compute direction (CG): use Hessian-vector products for several CG iterations to solve $\mathbf{s}\approx\mathbf{H}^{-1}\mathbf{g}$.
4. Compute coefficient: use the formula for maximum step-size coefficient $\beta$.
5. Line search: probe along $\beta\mathbf{s}$ to find the largest feasible step.
6. Update: $\theta_{\mathrm{new}}=\theta_{\mathrm{old}}+\mathrm{final\_step}$.

## References

- Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477). ICML 2015.
- Pearlmutter, B. A. (1994). [Fast Exact Multiplication by the Hessian](https://doi.org/10.1162/neco.1994.6.1.147). Neural Computation, 6(1), 147–160.
