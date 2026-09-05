---
title: "5.2 Momentum"
chapter_title: "Optimization Algorithms"
section_id: "05-02"
language: en
source_language: zh
source_docx: "第1部分 深度学习/5.优化算法/5.2 动量法（Momentum）.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 5.2 Momentum

## I. What Problems Does Momentum Address?

Momentum primarily addresses SGD's convergence speed and stability on certain landscapes.

1. Oscillations in narrow "ravines"

Description: imagine loss-function contours resembling a long, narrow valley. Gradients are large in the steep direction (the valley walls) but small in the gentle direction leading toward the optimal solution at the bottom.

SGD's problem: large gradients cause it to overshoot back and forth in the steep direction, resulting in very slow progress along the gentle direction.

2. Saddle points and flat regions

Description: local minima are relatively uncommon in high-dimensional spaces; saddle points are more common. At a saddle point, gradients in all directions are close to zero.

SGD's problem: as gradients approach 0, its update steps also approach zero, leaving the optimizer "stuck" and no longer updating.

## II. How Does Momentum Address These Problems?

Momentum introduces the physical concepts of "momentum" and "inertia." Imagine a ball rolling downhill: its motion depends not only on the local slope but also on velocity accumulated in the past (its "momentum").

In the steep direction, gradients $g_t$ alternate ($+C, -C, +C, -C, \ldots$). As they accumulate into momentum $m_t$, opposite signs cancel, reducing that component of $m_t$ and suppressing oscillations.

In the gentle direction, gradients $g_t$ consistently point the same way ($-c, -c, -c, \ldots$). Momentum $m_t$ accumulates these aligned gradients, increasing the component of $m_t$ in that direction and accelerating progress.

At a saddle point, the current gradient $g_t$ may be close to $0$, but as long as the ball carries past momentum ($m_{t-1}>0$), its "inertia" can carry it through the flat region toward the optimum.

## III. Mathematical Expression

Momentum introduces a hyperparameter β1 close to 1, representing the decay rate of "inertia." Instead of updating with the gradient directly, it uses the moving average "past gradient*β1 + current gradient*(1-β1)," multiplied by the learning rate.

Compute the gradient:

$$
g_t = \nabla_\theta J(\theta_{t-1})
$$

Update momentum (the first-moment estimate):

$$
m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1)\cdot g_t
$$

$m_t$ is the exponential moving average of $g_t$. $\beta_1$ controls how quickly past gradients are "forgotten." With $\beta_1=0.9$, approximately $90\%$ of current $m_t$ comes from previous momentum and $10\%$ from the current gradient.

Note: some classic implementations use $m_t=\beta_1\cdot m_{t-1}+g_t$. Adam uses the EMA form, which is used consistently here.

Update parameters:

$$
\theta_t = \theta_{t-1} - \alpha \cdot m_t
$$

The update direction now follows smoothed $m_t$, not $g_t$.

## References

- Polyak, B. T. (1964). [Some methods of speeding up the convergence of iteration methods](https://doi.org/10.1016/0041-5553(64)90137-5). USSR Computational Mathematics and Mathematical Physics.
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html). ICML 2013.
