---
title: "5.5 The AdamW Optimizer"
chapter_title: "Optimization Algorithms"
section_id: "05-05"
language: en
source_language: zh
source_docx: "第1部分 深度学习/5.优化算法/5.5 AdamW优化器.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 5.5 The AdamW Optimizer

AdamW (Adam with Decoupled Weight Decay) improves upon Adam.

## I. Motivation: Incorrect Scaling of Regularization by Adam's Adaptive Learning Rates

In conventional Adam, we add an $L_2$ penalty to the loss:

$$
\mathcal{L}_{total}(\theta) = f(\theta) + \frac{\lambda}{2}\lVert\theta\rVert^2
$$

The gradient $\hat{g}_t$ passed to the optimizer becomes:

$$
\hat{g}_t = \nabla f(\theta_{t-1}) + \lambda\theta_{t-1}
$$

Adam uses this regularized gradient $\hat{g}_t$ to compute $m_t$ and $v_t$:

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)\hat{g}_t
$$

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)\hat{g}_t^2
$$

The parameter update is:

$$
\theta_t = \theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

The problem: regularization term $\lambda\theta$ is mixed into $m_t$ and $v_t$. Adam's adaptive division by $\sqrt{v_t}$ therefore scales the regularization term too.

For parameters whose gradients vary greatly (large $v_t$), regularization is weakened. For parameters with gently varying gradients (small $v_t$), it is amplified. Different parameters consequently receive uneven decay, and the best $\lambda$ becomes strongly coupled with learning rate $\eta$, making tuning difficult.

We want to reduce the learning rate when the gradient of Loss with respect to parameters is too large and increase it when too small, without affecting regularization. The regularization term must therefore be decoupled.

## II. AdamW's Solution

AdamW removes weight decay from gradient $\hat{g}_t$. The gradient uses only the original loss:

$$
g_t = \nabla f(\theta_{t-1})
$$

Moment computations use only the original gradient:

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

The update has two parts: an Adam direction and independent weight decay:

$$
\theta_t = \theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta_{t-1}
$$

The first part is the standard Adam step and the second is decoupled weight decay.

Weight decay acts directly on parameters, no longer affected by learning-rate scaling. This restores its original physical meaning in SGD (shrinking weights proportionally every iteration), generally improving generalization over Adam.

## References

- Loshchilov, I., & Hutter, F. (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). ICLR 2019.
