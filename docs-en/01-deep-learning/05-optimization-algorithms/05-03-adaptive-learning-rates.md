---
title: "5.3 Adaptive Learning Rates"
chapter_title: "Optimization Algorithms"
section_id: "05-03"
language: en
source_language: zh
source_docx: "第1部分 深度学习/5.优化算法/5.3 自适应学习率.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 5.3 Adaptive Learning Rates

## I. What Problem Do Adaptive Learning Rates Address?

They primarily address SGD's learning-rate selection problem, especially when gradient scales differ greatly across parameters.

1. The global learning-rate dilemma

In deep networks, gradients for different parameters, such as shallow-layer weights, deep-layer weights, and biases, can differ by orders of magnitude. SGD uses one learning rate for all of them. If it is too large, parameters with large gradients (where the loss changes sharply) update too quickly, oscillating or even "exploding." If too small, parameters with small gradients (where the loss varies gently) update too slowly, requiring extremely long convergence times.

2. Nonstationary objectives

In some tasks, such as reinforcement learning, or later in training, gradient statistics such as average magnitude may change. A fixed value cannot accommodate this.

## II. The Adaptive Learning-Rate Solution

When updating parameters, an adaptive method divides each dimension's learning rate by the root mean square of its past gradients (sqrt(v_t)), scaling its magnitude. Parameters with consistently large recent gradients automatically receive smaller effective learning rates, preventing aggressive updates and oscillations. Those with consistently small recent gradients receive larger effective learning rates to converge faster.

Two approaches arise for computing this root mean square:

1. AdaGrad: use the arithmetic mean of all past squared gradients.

2. Root Mean Square Propagation (RMSProp): use a moving average v_t of past squared gradients. This vector, with the same dimensions as the parameters, tracks each parameter's recent average gradient magnitude (or energy), placing greater weight on recent gradients.

RMSProp performs better, so the following discussion focuses on it.

Two questions arise:

1. Why use recent history rather than one step?

The "overall landscape" better reflects reality. A single point may have a large gradient even when gradients are generally small. We should not divide by a large value in that case: first, a single point can be incidental; second, it may genuinely be special and should not be suppressed by "averaging."

2. Why use the root mean square?

First, violent oscillation within a "ravine" produces large gradients. The root mean square, highly sensitive to outliers, increases rapidly and sharply reduces that parameter's learning rate—exactly the desired behavior. We need a powerful automatic stabilizer that reacts quickly and strongly to oscillation (large gradients).

Second, as optimization proceeds, the mean gradient approaches 0 (with positive and negative values). Since variance is E(X^2)-E(X)^2, when E(X)≈0, the mean square effectively estimates variance. Statistically, variance represents the "degree of oscillation," making it suitable for adjusting learning-rate magnitude.

## III. Mathematical Expression

RMSProp introduces two hyperparameters for gradient division: β2 and epsilon. The former is the decay rate for the moving average of squared gradients; the latter is a tiny value such as 10^-8 that prevents a zero denominator. Updates divide gradients elementwise by (root mean square of recent gradients + epsilon), then multiply by the learning rate.

Compute the gradient:

$$
g_t = \nabla_\theta J(\theta_{t-1})
$$

Update the EMA of squared gradients (the second-moment estimate):

$$
v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2)\cdot (g_t \odot g_t)
$$

Note: $\odot$ denotes elementwise multiplication, so each element of $g_t$ is squared individually.

Update parameters:

$$
\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t}+\epsilon}\odot g_t
$$

The division and multiplication here are also elementwise. $\alpha$ is scaled by the "root mean square" corresponding to each parameter in $v_t$.

Note: RMSProp was first made public through Geoffrey Hinton's course slides, not an "original paper." The slides use $\frac{\alpha}{\sqrt{v_t}+\epsilon}\odot g_t$ in the update, not $m_t$.

## References

- Duchi, J., Hazan, E., & Singer, Y. (2011). [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html). Journal of Machine Learning Research.
- Hinton, G., Srivastava, N., & Swersky, K. (2012). [Neural Networks for Machine Learning, Lecture 6e: RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf). University of Toronto / Coursera.
