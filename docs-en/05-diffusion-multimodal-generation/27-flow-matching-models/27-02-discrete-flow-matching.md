---
title: "27.2 Discrete Flow Matching"
chapter_title: "Flow-Matching Models"
section_id: "27-02"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/27.流匹配模型/27.2 离散流匹配.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 27.2 Discrete Flow Matching

## I. Background

Current generative models (such as Stable Diffusion) are very successful in continuous spaces (where pixel values are continuous real numbers). Text, however, is discrete (token IDs are integers, such as 105 and 2099). There is no notion of a “small change” in a discrete space (changing 105 to 105.1 is meaningless), so directly applying continuous diffusion models is difficult.

## II. Core Idea

Continuous flow matching defines a vector field v_t(x) describing how a sample point x moves through space over time t, with differential equation dx/dt=v_t(x). In a discrete space, we cannot differentiate the discontinuously changing position x, but the probability distribution p_t(x) varies continuously with time, so we can model it with a continuous-time Markov chain (CTMC).

The authors found that, if we focus on the “path” that smoothly transitions from a noise distribution p to a real-data distribution q, and define a “probability velocity” describing how the probability distribution of x moves from p toward q, the discrete and continuous formulas share the same mathematical form. This allows many techniques that work well in continuous diffusion (such as scheduler design and denoising parameterization) to transfer directly.

## III. Mathematical Formulation

Note: in the expressions below, the number of possible sample values (the number of elements in the sample space) is finite, the probability values are discrete, and p denotes the actual probability of a sample, not a probability density function.

1. Probability path

Background: the goal of a generative model is to turn a simple random-noise distribution $p_0$ (such as all masks or random gibberish) into the real-data distribution $p_1$ (such as fluent sentences).

Interpolation $p_t$: we need to define a path $p_t$ that varies with time $t$ (from $0$ to $1$):

- At $t=0$, $p_0$ is entirely noise.
- At $t=1$, $p_1$ is entirely real data.
- This path defines the intermediate process: how the probability distribution $p_t$ is computed over $(0,1)$.

$$
p_t(x)=\sum_{x_0,x_1\in\mathcal{D}}p_t(x\mid x_0,x_1)\pi(x_0,x_1)
$$

This formula describes how the overall path consists of countless “individual paths.”
Here, pi(x_0,x_1) denotes “the probability that the initial state is x_0 and the final state is x_1”; p_t(x|x_0,x_1) denotes “the probability that the particular sample x appears at time t, given initial state x_0 and final state x_1” (x, x_0 and x_1 are three different samples here). This expression is therefore the law of total probability: summing the probabilities of obtaining vector x at time t under different (x_0,x_1) pairs gives the overall probability of sampling x at time t.

The “individual path” p_t(x|x_0,x_1) can itself be expressed as a convex combination of m probability distributions:

$$
p_t(x^i\mid x_0,x_1)=\sum_{j=1}^m\kappa_t^j w^j(x^i\mid x_0,x_1)
$$

The left-hand side, $p_t(x^i\mid x_0,x_1)$, means:

- This is the target conditional probability.
- It answers the question: “If we know that the starting point is $x_0$ (for example, the mask `[MASK]`) and the endpoint is $x_1$ (for example, the word ‘cat’), what is the probability that the token at position $i$ takes a particular value at some intermediate time $t$ (for example, at $50\%$ progress)?”

The right-hand $\sum_{j=1}^m$ means:

- This is a weighted sum, indicating that the final probability distribution is a mixture of $m$ basis distributions.

The basis distribution is denoted by $w^j(x^i\mid x_0,x_1)$ and means:

- It is a basic building block.
- Each $w^j$ represents a simple probabilistic behavior. For example:
  - $w^1$ might represent “remain at the starting point $x_0^i$.”
  - $w^2$ might represent “become the endpoint $x_1^i$.”
  - $w^3$ might represent “become completely random noise.”

Thus, the formula defines a mixture of probabilities: at time $t$, this position follows distribution $w^1$ with probability $\kappa_t^1$, follows distribution $w^2$ with probability $\kappa_t^2$, and so on.

For point x_i, the final probability distribution is a weighted sum (convex combination) of these m basis distributions. If m=2:

1. $w^1$ is the probability of “retaining the initial noise $x_0$” ($\delta_{x_0}$).
2. $w^2$ is the probability of “becoming the final data $x_1$” ($\delta_{x_1}$).

The formula becomes:

$$
p_t(x^i\mid x_0,x_1)=(1-\kappa_t)\delta_{x_0^i}+\kappa_t\delta_{x_1^i}
$$

**Physical interpretation**: at time $t$, the word at this position has probability $\kappa_t$ of already being the final word $x_1$ and probability $1-\kappa_t$ of remaining the initial noise word $x_0$. As time $t$ advances, $\kappa_t$ gradually increases, and the whole sequence slowly “flows” from noise toward real data.

This is equivalent to m=2 meaning “select a point on the line segment between w_1 and w_2,” while m=3 means “select a point inside the triangle formed by w_1, w_2 and w_3,” and so on. Geometrically:

Suppose our vocabulary contains only three words: cat, dog and bird, so $d=3$.

- **Discrete state**: any definite word (such as “cat”) can be represented by a one-hot vector: $[1,0,0]$.
- **Probability distribution**: the model does not output a definite word, but a probability distribution such as $[0.7,0.2,0.1]$ ($70\%$ cat, $20\%$ dog, and so on).
- **Simplex**: the set of vectors whose elements are all nonnegative and sum to $1$ geometrically forms a simplex.
  - For three words, the simplex is an equilateral triangle in three-dimensional space.
  - Its three vertices are $[1,0,0]$, $[0,1,0]$ and $[0,0,1]$, representing definite words.
  - Any point inside the triangle represents an intermediate probability state.

$$
p_t(x^i\mid x_0,x_1)=\sum_{j=1}^{m}\kappa_t^{i,j}w^j(x^i\mid x_0,x_1)
$$

Here, $\sum \kappa=1$ and $\kappa\ge 0$, which is precisely the geometric definition of a “convex combination.” This means that, regardless of how the basis paths $w^j$ are designed, the generated probability distribution $p_t$ never leaves the simplex as long as this formula is satisfied. It ensures that the model always produces valid probability distributions.

2. Generating probability velocities

**The continuity equation**: to define discrete flow matching, we use the CTMC paradigm. Samples $X_t$ jump between discrete states. We need to predict the **rate** at which each token's probability changes. The continuity equation describes the relationship between the rate of change in state probability $\dot{p}_t(x)$ and the flux:

$$
\dot{p}_t(x)+\mathrm{div}_x(p_tu_t)=0
$$

Here, $\mathrm{div}_x$ is the divergence operator. In the discrete case, divergence is defined as outgoing flux minus incoming flux:

$$
\mathrm{div}_x(v)=\sum_{z\in\mathcal{D}}[v(z,x)-v(x,z)]
$$

**Theorem 2 (marginal velocity)**: given a conditional probability velocity $u_t^i(x^i,z\mid x_0,x_1)$ that generates the conditional probability path $p_t(x\mid x_0,x_1)$, its marginal velocity $u_t$ is:

$$
u_t^i(x^i,z)=\sum_{x_0,x_1\in\mathcal{D}}u_t^i(x^i,z\mid x_0,x_1)p_t(x_0,x_1\mid z)
$$

This follows the same logic as continuous flow matching: the marginal vector field is the posterior expectation of the conditional vector field.

**Theorem 3 (closed-form solution)**: for the path defined by Eq. 9 (the linear interpolation path), the marginal generating probability velocity has a very concise closed-form solution that closely matches the form of continuous flow matching:

$$
u_t^i(x^i,z)=\frac{\dot{\kappa}_t}{1-\kappa_t}[p_{1\mid t}(x^i\mid z)-\delta_z(x^i)]
$$

Here, $p_{1\mid t}(x^i\mid z)=\mathbb{E}[X_1\mid X_t=z]$ is the **probability denoiser**. This means that the velocity field can be parameterized by learning a neural network that predicts the distribution of the target $X_1$.

3. Training objective

**The power of Theorem 3**: it turns complex “velocity computation” into a simple supervised learning task. The formula $u_t\propto p_{1\mid t}-\delta_z$ tells us that we only need to train a neural network that takes the current noisy sentence $z$ and time $t$ as input and predicts the probability distribution $p_{1\mid t}$ of the original clean sentence $X_1$. This network then automatically defines the correct “flow” velocity.

**Training loss**: based on the derivation above, the training objective simplifies to minimizing cross-entropy loss:

$$
\mathcal{L}(\theta)=-\sum_i\mathbb{E}_{t,data}\log p_{1\mid t}(X_1^i\mid X_t;\theta)
$$

This is the standard **denoising** objective: show the model noisy/masked data $X_t$ and ask it to predict the original data $X_1$. This explains why BERT-style masked language modeling (MLM) is, in fact, training a flow model.

There is, of course, one crucial difference from BERT's MLM training: the masking rate is dynamic. At t=0, almost everything is masked, whereas at t=1, almost everything is real data. The model's input is not only noisy data X_t, but (X_t,t), meaning it must learn to recover the original data X_1 at different noise levels (corresponding to time t).

## References

- Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R. T. Q., Synnaeve, G., Adi, Y., & Lipman, Y. (2024). [Discrete Flow Matching](https://arxiv.org/abs/2407.15595). arXiv:2407.15595.
- Campbell, A., Yim, J., Barzilay, R., Rainforth, T., & Jaakkola, T. (2024). [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997). arXiv:2402.04997.
