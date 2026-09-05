---
title: "26.1 Basic Principles of Diffusion Models"
chapter_title: "Diffusion Models"
section_id: "26-01"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/26.扩散模型/26.1 扩散模型的基本原理.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 26.1 Basic Principles of Diffusion Models

## I. Forward Diffusion

Given a data point sampled from the real data distribution, we define a forward diffusion process that adds small amounts of Gaussian noise to the sample in successive steps, producing a sequence of noisy samples. The step size is controlled by a variance schedule. As the step size increases, the data sample gradually loses its distinguishing features. Eventually, it becomes equivalent to an isotropic Gaussian distribution.

The forward process is a fixed process that systematically and gradually corrupts a clear original image $x_0$ into pure Gaussian noise.

Mathematically, this process is defined as a Markov chain, in which each step depends only on the preceding step. At each step $t$, we add Gaussian noise to the previous state $x_{t-1}$ to obtain:

$$
x_t = \sqrt{\alpha_t}\cdot x_{t-1} + \sqrt{1-\alpha_t}\cdot \epsilon_t
$$

Here, $\epsilon_t \sim \mathcal{N}(0, I)$ is standard Gaussian noise, $\alpha_t = 1-\beta_t$, and $\beta_t$ is a predefined noise-scheduling parameter that increases with time $t$ and controls the amount of noise added at each step.

Simply put, using $\sqrt{\alpha_t}$ and $\sqrt{1-\alpha_t}$ rather than $\alpha_t$ and $1-\alpha_t$ directly preserves the variance of the data, or prevents its “energy” from dissipating or exploding as noise is repeatedly added.

By the linearity of expectation:

$$
\mathbb{E}[x_t]
= \mathbb{E}\left[\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon\right]
= \sqrt{\alpha_t}\underbrace{\mathbb{E}[x_{t-1}]}_{0}
+ \sqrt{1-\alpha_t}\underbrace{\mathbb{E}[\epsilon]}_{0}
= 0
$$

Whether the coefficient is $\sqrt{\alpha}$, $\alpha$, or even some arbitrary coefficient, the resulting mean is always $0$ as long as both the input and the noise have mean $0$.

Thus, the mean is not the issue and requires no special attention.

$$
\begin{aligned}
x_t
&= \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1},
\quad \text{where } \epsilon_{t-1},\epsilon_{t-2},\cdots \sim \mathcal{N}(0,I) \\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2}
 + \sqrt{1-\alpha_t\alpha_{t-1}}\bar{\epsilon}_{t-2},
\quad \text{where } \bar{\epsilon}_{t-2} \text{ merges two Gaussians } (*) \\
&= \cdots \\
&= \sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
\end{aligned}
$$

(*) indicates the combination of two Gaussian distributions; the mathematical derivation is omitted here.

Here, $\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$ is the cumulative product of $\alpha_s$, and $\epsilon\sim\mathcal{N}(0,I)$. This formula greatly simplifies the process because it allows us to obtain any intermediate noisy state directly, without iterating step by step. When $t$ is sufficiently large (for example, $T=1000$), $\bar{\alpha}_t$ approaches $0$, and $x_T$ becomes almost pure noise $\epsilon$.

The probabilistic representation of each recursive forward-diffusion step is:

$$
q(x_t\mid x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)
$$

The right-hand side means that $x_t$ is obtained from a Gaussian distribution with mean $\sqrt{1-\beta_t}x_{t-1}$ and covariance $\beta_t I$.

## II. Reverse Diffusion

1. Mathematical derivation

In the reverse process of a diffusion model, our goal is to progressively recover real data $x_0$ from pure noise $x_T$. In theory, the process of working backward from $x_t$ to $x_{t-1}$ is a conditional probability distribution $p_\theta(x_{t-1}\mid x_t)$.

One of the useful properties of diffusion models is that, when the total number of steps T is sufficiently large, this reverse conditional probability can be approximated by a Gaussian distribution:

$$
p(x_{t-1}\mid x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$

There are two key parameters here: the mean $\mu_\theta$ and the variance $\Sigma_\theta$.

- The variance $\Sigma_\theta(x_t,t)$ is usually set to a constant $\sigma_t^2 I$ associated with the time step $t$, and does not need to be learned by the network.
- The fundamental task of the neural network is therefore to predict the mean $\mu_\theta(x_t,t)$ of this Gaussian distribution.

In other words, the diffusion model first predicts a deterministic noise value, then uses it to obtain the probability distribution (a Gaussian distribution) of the denoised $x_{t-1}$, and samples $x_{t-1}$ from that distribution (this is where randomness enters). We derive this probability distribution below.

Step 1: Apply Bayes' theorem

Our goal is to find the true reverse transition probability $q(x_{t-1}\mid x_t,x_0)$ when the original image $x_0$ is known. Computing it directly is difficult, but Bayes' theorem allows us to decompose it into several known distributions:

$$
q(x_{t-1}\mid x_t,x_0)
= \frac{q(x_t\mid x_{t-1},x_0)\cdot q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

By the Markov property of the diffusion model, the current state depends only on the preceding state, so $q(x_t\mid x_{t-1},x_0)=q(x_t\mid x_{t-1})$. This distribution is precisely the definition of the forward process:

$$
q(x_t\mid x_{t-1})=\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)
$$

Meanwhile, $q(x_{t-1}\mid x_0)$ and $q(x_t\mid x_0)$ are closed-form solutions of the forward process, representing the distributions after $t-1$ and $t$ noise-addition steps starting from $x_0$:

- $q(x_t\mid x_{t-1})=\mathcal{N}(x_t\mid\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)$ represents noise addition from $x_{t-1}$ to $x_t$.
- $q(x_{t-1}\mid x_0)=\mathcal{N}(x_{t-1}\mid\sqrt{\bar{\alpha}_{t-1}}x_0,(1-\bar{\alpha}_{t-1})I)$, because $x_{t-1}$ is obtained by accumulating noise over $t-1$ steps from $x_0$.
- $q(x_t\mid x_0)=\mathcal{N}(x_t\mid\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)$, because $x_t$ is obtained by accumulating noise over $t$ steps from $x_0$.

Substituting into Bayes' formula gives:

$$
q(x_{t-1}\mid x_t,x_0)
= \frac{
\mathcal{N}(x_t\mid\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)\cdot
\mathcal{N}(x_{t-1}\mid\sqrt{\bar{\alpha}_{t-1}}x_0,(1-\bar{\alpha}_{t-1})I)
}{
\mathcal{N}(x_t\mid\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
}
$$

We have now decomposed an unknown distribution into three known Gaussian distributions.

Step 2: Combine the Gaussian distributions

We want to show that this ratio also follows a Gaussian distribution. We can ignore the normalization constant and focus only on the exponent of the probability density function, because the exponent of a Gaussian distribution is quadratic. Thus:

$$
q(x_{t-1}\mid x_t,x_0)
\propto
\exp\left\{
-\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{2(1-\alpha_t)}
-\frac{(x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0)^2}{2(1-\bar{\alpha}_{t-1})}
+\frac{(x_t-\sqrt{\bar{\alpha}_t}x_0)^2}{2(1-\bar{\alpha}_t)}
\right\}
$$

Define the quadratic function:

$$
f(y)=
\frac{(x-\sqrt{a}y)^2}{2(1-a)}
+\frac{(y-\sqrt{b}z)^2}{2(1-b)}
-\frac{(x-\sqrt{c}z)^2}{2(1-c)}
$$

The mapping is:

- $x=x_t$, $a=\alpha_t$;
- $y=x_{t-1}$, $b=\bar{\alpha}_{t-1}$; note that $b$ uses the cumulative parameter $\bar{\alpha}_{t-1}$ rather than $\alpha_{t-1}$;
- $z=x_0$, $c=\bar{\alpha}_t$.

Since $f(y)$ is quadratic, its minimizer is the mean of the Gaussian distribution.

Differentiate $f(y)$:

$$
f'(y)=-\frac{\sqrt{a}}{1-a}(x-\sqrt{a}y)+\frac{1}{1-b}(y-\sqrt{b}z)
$$

Expand:

$$
f'(y)=\frac{a}{1-a}y-\frac{\sqrt{a}}{1-a}x+\frac{1}{1-b}y-\frac{\sqrt{b}}{1-b}z
$$

Collect the terms in $y$:

$$
f'(y)=\left(\frac{a}{1-a}+\frac{1}{1-b}\right)y-\frac{\sqrt{a}}{1-a}x-\frac{\sqrt{b}}{1-b}z
$$

Set $f'(y)=0$:

$$
y=\frac{\frac{\sqrt{a}}{1-a}x+\frac{\sqrt{b}}{1-b}z}{\frac{a}{1-a}+\frac{1}{1-b}}
$$

Simplify the denominator:

$$
\frac{a}{1-a}+\frac{1}{1-b}
=\frac{a(1-b)+(1-a)}{(1-a)(1-b)}
=\frac{1-ab}{(1-a)(1-b)}
$$

The numerator is:

$$
\frac{\sqrt{a}}{1-a}x+\frac{\sqrt{b}}{1-b}z
$$

Therefore:

$$
y=\frac{(1-b)\sqrt{a}x+(1-a)\sqrt{b}z}{1-ab}
$$

Substitute the mapping:

- $a=\alpha_t$;
- $b=\bar{\alpha}_{t-1}$;
- $c=\bar{\alpha}_t$;
- $ab=\alpha_t\cdot\bar{\alpha}_{t-1}=\bar{\alpha}_t$, because $\bar{\alpha}_t=\alpha_t\cdot\bar{\alpha}_{t-1}$.

Substitution gives:

$$
y=\frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}x_t+(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_t}
$$

This is exactly the mean given in the theorem:

$$
\mu_q(x_t,x_0)=
\frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1-\bar{\alpha}_t}x_t
+\frac{(1-\alpha_t)\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_t}x_0
$$

Compute the second derivative:

$$
f''(y)=\frac{a}{1-a}+\frac{1}{1-b}
=\frac{a(1-b)+(1-a)}{(1-a)(1-b)}
=\frac{1-ab}{(1-a)(1-b)}
$$

Substitute:

- $a=\alpha_t$;
- $b=\bar{\alpha}_{t-1}$;
- $ab=\bar{\alpha}_t$.

Then:

$$
f''(y)=\frac{1-\bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}
$$

The variance is the reciprocal:

$$
\Sigma_q(t)=\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}I
$$

The final result is therefore:

$$
\tilde{\mu}_t=
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t}x_0
$$

$$
\tilde{\beta}_t=\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
$$

This formula shows that, given the original image $x_0$, the mean of the reverse distribution is a linear combination of the current noisy image $x_t$ and the original image $x_0$.

Step 3: Rewrite the expression through parameterization

The mean formula above still depends on the unknown $x_0$. This is where the reparameterization trick for the forward process becomes crucial. The forward process tells us that $x_t$ at any time can be expressed directly in terms of $x_0$ and a small noise term $\epsilon$:

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

We can solve this equation for $x_0$:

$$
x_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
$$

Now substitute the expression for $x_0$ into the mean formula obtained in Step 2:

$$
\tilde{\mu}_t=
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t}
\left(\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}\right)
$$

The next step is to combine like terms and simplify. Some algebra yields an extremely concise form:

$$
\tilde{\mu}_t=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)
$$

Here, $\epsilon$ represents the “total equivalent noise” added over the steps from $x_0$ to $x_t$.
When the real data is fixed, the variance predicted at this step is still determined by $\alpha_t=1-\beta_t$:

$$
\tilde{\mu}_t=
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t}x_0
$$

$$
\tilde{\beta}_t=\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
$$

This formula shows that, given the original image $x_0$, the mean of the reverse distribution is a linear combination of the current noisy image $x_t$ and the original image $x_0$.
For further discussion of variance, see the section on Diffusion Transformer.

2. Application in the model

The formulas above tell us that, if we know the noise $\epsilon_t$ added at step $t$ in the forward process, we can calculate the mean of the reverse step exactly and thereby recover the distribution of $x_{t-1}$ from $x_t$. The problem, however, is that we only know the noise follows a Gaussian distribution $\mathcal{N}(0,I)$; we know nothing about the specific value sampled from that distribution that turned the original $x_0$ into the noisy $x_t$.

Since we cannot know $\epsilon$ in advance, we train a neural network $\epsilon_\theta$ to predict it. The training objective is to minimize the mean squared loss (difference) between the predicted noise $\epsilon_\theta(x_t,t)$ and the actual noise $\epsilon$. This turns the complex problem of predicting a distribution into the more stable problem of predicting noise, substantially reducing the difficulty of learning.

This setup is very similar to a variational autoencoder (VAE), so we can use a variational lower bound to optimize the negative log-likelihood.

## III. Training Diffusion Models

1. Randomly sampling the starting denoising time step during training

In the theoretical framework of diffusion models, the generation process (reverse denoising) is a Markov chain that must move backward step by step from $t=T$ to $t=0$.

If training also proceeded sequentially in the order $T,T-1,\ldots,0$, it would create two critical problems:

1. **Extremely high computational cost**: each backward pass would have to traverse the entire time chain, and gradient accumulation would exhaust GPU memory.
2. **Cascading errors**: if the earlier steps were not learned well, the later steps would drift increasingly far from the target because they would operate on incorrect inputs.

To solve this problem, the authors of Denoising Diffusion Probabilistic Models (DDPM) used the “additivity” of Gaussian distributions (the reparameterization trick) to derive a remarkable result: **we do not need to add noise step by step; instead, we can write down the analytical solution for the state at any time $t$ directly**.

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

With this formula, the model's final optimization objective (a simplified version of the evidence lower bound, ELBO) can be written as an expectation over all time steps $t$:

$$
\mathcal{L}_{simple}
=\mathbb{E}_{t\sim\mathcal{U}(1,T),\,x_0\sim q(x_0),\,\epsilon\sim\mathcal{N}(0,I)}
\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2\right]
$$

This is the mathematical core: the outer $\mathbb{E}_{t\sim\mathcal{U}(1,T)}$ means that the loss is an expectation over uniformly distributed $t$. In actual training code, this global expectation is computed through Monte Carlo sampling: a time step $t$ is sampled uniformly at random in each batch.

2. Parallelism

Each step of a diffusion model denoises all tokens simultaneously, so the processing is parallel. From a probabilistic perspective:

Suppose we want to generate a sentence:

$$
X = \{x_1, x_2, x_3, x_4\}
$$

An autoregressive model strictly follows the probability chain rule:

$$
P(X) = P(x_1)P(x_2 \mid x_1)P(x_3 \mid x_1, x_2)P(x_4 \mid x_1, x_2, x_3)
$$

To compute $x_3$, definite values of $x_1,x_2$ must already be available.

By contrast, the goal of a diffusion model is to model the joint distribution $P(x_1,x_2,x_3,x_4)$ directly. At a generation step (for example, denoising from $X^{(T+1)}$ to $X^{(T)}$), the model usually assumes that predictions at different positions are conditionally independent at the current step; that is, $x_1^{(T)},x_2^{(T)},\ldots,x_N^{(T)}$ are mutually independent. We infer the current $x_i^{(T)}$ only from the previous step's global context $x_1^{(T+1)},x_2^{(T+1)},\ldots,x_N^{(T+1)}$, filling in all details simultaneously from the “blurry global outline.” Thus, in text diffusion, the formula is often approximated as:

$$
\begin{aligned}
P(X^{(T)} \mid X^{(T+1)})
&\approx
P(x_1^{(T)} \mid x_1^{(T+1)},x_2^{(T+1)},\ldots) \\
&\quad \cdot P(x_2^{(T)} \mid x_1^{(T+1)},x_2^{(T+1)},\ldots) \\
&\quad \cdots
P(x_N^{(T)} \mid x_1^{(T+1)},x_2^{(T+1)},\ldots)
\end{aligned}
$$

This means that, within one prediction step, guessing $x_2$ does not require waiting for $x_1$ to be denoised first, improving GPU utilization.

For tasks such as text generation, diffusion models produce lower-quality outputs than autoregressive models. Although a diffusion model can generate 10 words at once in parallel, it assumes within a single generation step that these words are generated independently in parallel (or depend only on the previous round's blurry state). It therefore often overlooks subtle, strong causal relationships between words. The autoregressive approach is: “Because the first word is ‘artificial,’ the second word is probably ‘intelligence.’” (Strong causality.) The purely parallel approach is: “The overall context seems to be a technology topic, so I guess ‘artificial’ at position 1 and ‘intelligence’ at position 2.” Excessive parallelism can sometimes create logical inconsistencies and is liable to cause breaks in logic in more complex reasoning tasks.

3. Training algorithm workflow

By sampling $t$ randomly, a diffusion model elegantly decomposes a sequential temporal generation problem into independent denoising tasks that can be trained entirely in parallel. The specific algorithmic workflow is:

- **Step 1: Sample real data.** Randomly select a clean real image $x_0\sim q(x_0)$ from the training set.
- **Step 2: Randomly sample a time step.** Randomly select a time step $t\sim\mathcal{U}(1,T)$ from a uniform distribution (usually $T=1000$).
- **Step 3: Sample standard Gaussian noise.** Generate a random noise matrix $\epsilon\sim\mathcal{N}(0,I)$ with exactly the same dimensions as $x_0$.
- **Step 4: Construct the noisy image in one step.** Use the closed-form solution derived above, $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$, to compute the image's noisy state $x_t$ at time $t$ directly.
- **Step 5: Predict with the neural network.** Feed the noisy image $x_t$ and the time step $t$ (usually represented through sinusoidal positional encoding) into the neural network $\epsilon_\theta$ (such as U-Net or DiT), and let it predict the noise just added, $\epsilon_\theta(x_t,t)$.
- **Step 6: Compute the loss and update gradients.** Compute the mean squared error (MSE) between the actual noise $\epsilon$ and the predicted noise $\epsilon_\theta$, and perform backpropagation to update the network weights.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456). ICLR.
- Song, J., Meng, C., & Ermon, S. (2021). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). ICLR.
