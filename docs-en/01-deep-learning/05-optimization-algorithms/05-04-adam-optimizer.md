---
title: "5.4 The Adam Optimizer"
chapter_title: "Optimization Algorithms"
section_id: "05-04"
language: en
source_language: zh
source_docx: "第1部分 深度学习/5.优化算法/5.4 Adam优化器.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 5.4 The Adam Optimizer

## I. What Problems Does Adam Address?

Adam addresses optimization difficulties using both momentum's "inertia" to accelerate convergence, suppress oscillations, and traverse saddle points, and RMSProp's adaptive learning rates to assign suitable update steps to different parameters.

## II. How Does Adam Address Them?

1. It combines both approaches, tracking exponential moving averages of two "moments":

First moment: the average gradient $m_t$ (the momentum component), determining update direction and momentum.

Second moment: the average squared gradient $v_t$ (the RMSProp component), adaptively scaling each parameter's learning rate as in RMSProp.

2. Key improvement: bias correction

Adam's authors identified a problem: $m_t$ and $v_t$ are both initialized to $0$ at $t=0$. Because $\beta_1$ and $\beta_2$ are close to $1$ (such as $0.9$ and $0.999$), $m_t$ and $v_t$ are strongly biased toward $0$ during the first few training steps. The authors therefore rescaled the expressions for $m_t$ and $v_t$ according to time.

## III. Mathematical Expression

Hyperparameters are $\alpha$ (learning rate), $\beta_1$ (such as $0.9$), $\beta_2$ (such as $0.999$), and $\epsilon$ (such as $10^{-8}$). Initialize $m_0=0$ (first-moment vector), $v_0=0$ (second-moment vector), and $t=0$.

At time step $t$:

1. Set $t \leftarrow t+1$.
2. Compute the gradient:

$$
g_t = \nabla_\theta J(\theta_{t-1})
$$

3. Update the biased first moment (momentum):

$$
m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1)\cdot g_t
$$

4. Update the biased second moment (RMSProp):

$$
v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2)\cdot (g_t \odot g_t)
$$

5. Compute the bias-corrected first moment:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

For small $t$, $1-\beta_1^t$ is close to $0$, amplifying $m_t$.

6. Compute the bias-corrected second moment:

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

For large $t$, $\beta^t \to 0$ and the denominator $\to 1$, so the correction disappears.

7. Update parameters:

$$
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

## IV. Applications

Adam and variants such as AdamW are the default first-choice optimizers for most deep learning tasks today.

1. Language models: an embedding layer takes text passages and produces integrated vector representations for subsequent processing. For example, a vocabulary of 50,000 words with 300-dimensional vectors requires 15,000,000 parameters. We update these parameters so that correct convergence—each embedding effectively representing meaning—produces correct representations of input passages. Any sentence, however, contains only a few words, activating and updating only their parameters. Common words therefore receive frequent gradient updates, while rare words such as "aardvark" receive very few. SGD overtrains common words and undertrains rare ones. Adam's adaptive rates perfectly address this: rare-word gradients remain $0$ for long periods, making $v_t$ very small as it decays exponentially toward $0$. When the rare word finally occurs, its nonzero $g_t$ is divided by a tiny value, producing a huge effective learning rate and an adequate update.

2. Large, complex models: large models can contain hundreds of billions or even trillions of parameters. Adam's momentum navigates these complex, high-dimensional loss spaces smoothly, avoiding becoming "stuck" at saddle points or flat regions and greatly accelerating convergence.

3. Building a new computer-vision model: SGD with momentum may require a carefully designed and tuned learning-rate schedule to converge. Adam is less sensitive to the initial learning-rate choice and usually produces very good results quickly with defaults such as α=0.001.

4. Generative-model training: GAN training, for example, is a "two-player zero-sum game" between generator and discriminator, requiring a delicate balance. If one trains too quickly, the other collapses and training fails. Adam's adaptive rates help automatically balance their training speeds, preventing exploding or vanishing gradients on either side and greatly improving GAN-training success. Diffusion models are also enormous and expensive to train, relying on Adam for stable, efficient training over massive datasets.

5. Reinforcement learning: agents change their policies as they learn, continually changing the distribution of collected data ("experience"). Rewards are often sparse and delayed, producing high-variance, noisy gradients. Adam's second moment $v_t$ (RMSProp) effectively estimates gradient variance, while its first moment $m_t$ (momentum) smooths noise and improves robustness. This makes it ideal for stable training in noisy, nonstationary RL environments.

## References

- Kingma, D. P., & Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). ICLR 2015.
