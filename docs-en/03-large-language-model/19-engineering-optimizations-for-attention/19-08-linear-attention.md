---
title: "19.8 Linear Attention"
chapter_title: "Engineering Optimizations for Attention"
section_id: "19-08"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.8 线性注意力（Linear Attention）.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.8 Linear Attention

## I. Basic Principles

The core idea of linearized attention is to avoid explicitly calculating and storing the $n \times n$ attention matrix $A$ by changing the order of computation.

Its objective is to reduce computational complexity from $O(n^2)$ to $O(n)$.

How is this achieved? The answer is to use the associativity of matrix multiplication.

- Standard computation order: $(QK^T)V$. First calculate the $n \times n$ matrix, then multiply it by $V$ ($n \times d$).
- Linearized computation order: $Q(K^T V)$. First calculate $K^T V$ (obtaining a $d \times d$ matrix), then multiply it by $Q$ ($n \times d$).

Linearization is not unconditional: Softmax must be removed, because:

Standard attention is:

$$
\mathrm{Output} = \mathrm{Softmax}(QK^T)V
$$

Notice that this contains not only multiplication, but also a nonlinear Softmax wrapped around the intermediate result.

$$
\mathrm{Softmax}(QK^T) \ne Q \cdot \mathrm{Softmax}(K^T)
$$

You cannot split Softmax or move it to the end in order to calculate $K^T V$ first. Every element of $Q$ and $K$ must first interact, be summed, and be normalized before multiplication by $V$. This means that the enormous $n \times n$ matrix must be constructed.

We need a kernel function to approximate Softmax. The mathematical derivation follows.

First, write standard attention in a row-normalized form. For the output $o_i$ of the $i$-th query vector $q_i$:

$$
o_i =
\frac{
\sum_{j=1}^{n} \exp\left(\frac{q_i k_j^T}{\sqrt{d_k}}\right)v_j
}{
\sum_{j=1}^{n} \exp\left(\frac{q_i k_j^T}{\sqrt{d_k}}\right)
}
$$

We can abstract this into a more general similarity function $\mathrm{sim}(\cdot,\cdot)$:

$$
o_i =
\frac{
\sum_{j=1}^{n} \mathrm{sim}(q_i,k_j)v_j
}{
\sum_{j=1}^{n} \mathrm{sim}(q_i,k_j)
}
$$

Next, introduce a kernel function $\phi$ and represent similarity as an inner product after feature mapping:

$$
\mathrm{sim}(q_i,k_j) = \phi(q_i)\cdot\phi(k_j)^T
$$

Here, $\phi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d_{proj}}$ maps the original vector into another feature space and is also called a kernel function.

Now substitute this kernel function into the calculation of output $o_i$:

$$
o_i =
\frac{
\sum_{j=1}^{n}[\phi(q_i)\cdot\phi(k_j)^T]v_j
}{
\sum_{j=1}^{n}[\phi(q_i)\cdot\phi(k_j)^T]
}
$$

Because $\phi(q_i)$ is constant with respect to the summation index $j$, we can move it outside the summation:

$$
o_i =
\frac{
\phi(q_i)\cdot\left[\sum_{j=1}^{n}\phi(k_j)^T \otimes v_j\right]
}{
\phi(q_i)\cdot\left[\sum_{j=1}^{n}\phi(k_j)^T\right]
}
$$

Here, $\otimes$ denotes the outer product; after fixing the row and column orientations of the vectors, it is also often written as matrix multiplication.

Note: an inner product multiplies a row vector by a column vector and yields a scalar; an outer product multiplies a column vector ($m \times 1$) by a row vector ($1 \times n$) and yields a matrix ($m \times n$).

Let us define two key accumulated-state matrices:

1. $S$: the aggregated state of keys and values.

$$
S = \sum_{j=1}^{n}\phi(k_j)^T v_j \in \mathbb{R}^{d_{proj}\times d_v}
$$

Note: $\phi(k_j)$ here is a row vector. After transposition, it is a column vector; its outer product with $v_j$ (a row vector) yields a matrix. We sum these matrices over all $j$.

2. $Z$: the aggregated state of the normalization factor.

$$
Z = \sum_{j=1}^{n}\phi(k_j)^T \in \mathbb{R}^{d_{proj}\times 1}
$$

Now, row $i$, $o_i$, of the output $O$ for the entire sequence can be calculated as:

$$
o_i = \frac{\phi(q_i)S}{\phi(q_i)Z}
$$

The denominator $\phi(q_i)Z$ is a scalar (the normalization factor), so every component of the output vector is divided by the same normalization value.

- Calculate $S$ and $Z$: traverse each position $j$ in the sequence, calculate $\phi(k_j)^T v_j$ and $\phi(k_j)^T$, and accumulate them. This step has complexity $O(n\cdot d_{proj}\cdot d_v)$.
- Calculate all outputs $o_i$: for each position $i$, calculate $\phi(q_i)S$ and $\phi(q_i)Z$. This step also has complexity $O(n\cdot d_{proj}\cdot d_v)$.

The total complexity is $O(nd_{proj}d_v)$. If $d_{proj}$ and $d_v$ are fixed constants independent of sequence length $n$, the complexity is $O(n)$, that is, linear.

Note: $K$ and $V$ originally have dimensions $n \times d_k$ and $n \times d_v$, respectively. Multiplying transposed $K$ by $V$ has matrix-multiplication complexity $n\cdot d_k\cdot d_v$, which is $O(n)$ with respect to sequence length. Here, $\phi$ changes only the feature dimension, not the linear structure of accumulation over sequence positions.

## II. How Can We Find a Kernel Function $\phi(x)$?

For the exponential similarity in Softmax, we want to find a feature mapping such that:

$$
\mathrm{sim}(q,k) = \exp(q\cdot k^T) \approx \phi(q)\cdot\phi(k)^T
$$

When using Softmax directly in deep learning models (such as the original Transformer attention mechanism), we perform numerical computation. For a single numerical value $x$, $\exp(x)$ is a well-established mathematical function that computers can handle efficiently. But when $q$ and $k$ are vectors, expressing $\exp(q\cdot k)$ as an inner product of two feature vectors naturally leads to a higher-dimensional feature mapping.

Start with the Taylor expansion:

$$
\exp(q\cdot k)
= 1 + (q\cdot k) + \frac{(q\cdot k)^2}{2!}
+ \frac{(q\cdot k)^3}{3!}
+ \frac{(q\cdot k)^4}{4!}
+ \cdots
$$

For $m=1$:

$$
(q\cdot k)^1 = \sum_{i=1}^{d}q_i k_i
$$

This can be regarded as the inner product of vectors $q$ and $k$ in the original $d$-dimensional space.

For $m=2$:

$$
(q\cdot k)^2
= \left(\sum_{i=1}^{d}q_i k_i\right)^2
= \sum_{i=1}^{d}\sum_{j=1}^{d}q_i q_j k_i k_j
$$

Notice that this can be rewritten as the inner product of two new vectors:

- Let $\phi_2(q)$ be a $d^2$-dimensional vector containing all combinations of $q_i q_j$.
- Let $\phi_2(k)$ be a $d^2$-dimensional vector containing all combinations of $k_i k_j$.

Then:

$$
(q\cdot k)^2 = \phi_2(q)\cdot\phi_2(k)
$$

We can now define a feature mapping for every power $m$:

- For $m=0$: $\phi_0(q)=1$, with 1 dimension.
- For $m=1$: $\phi_1(q)=q$, with $d$ dimensions.
- For $m=2$: $\phi_2(q)$ contains all $q_i q_j$, with $d^2$ dimensions.
- For $m=3$: $\phi_3(q)$ contains all $q_i q_j q_k$, with $d^3$ dimensions.
- And so on.

Combining these features of different orders constructs an infinite-dimensional super-feature mapping $\Phi(q)$:

$$
\Phi(q)=
\left(
1,\;
q,\;
\frac{q_i q_j}{\sqrt{2!}},\;
\frac{q_i q_j q_k}{\sqrt{3!}},\;
\frac{q_i q_j q_k q_l}{\sqrt{4!}},\;
\cdots
\right)
$$

Each component is divided by the corresponding $\sqrt{m!}$ to match the coefficients of the Taylor expansion.

Thus:

$$
\begin{aligned}
\Phi(q)\cdot\Phi(k)
&= 1\cdot 1
+ \sum_i q_i k_i
+ \sum_{i,j}\frac{q_i q_j}{\sqrt{2!}}\frac{k_i k_j}{\sqrt{2!}}
+ \sum_{i,j,k}\frac{q_i q_j q_k}{\sqrt{3!}}\frac{k_i k_j k_k}{\sqrt{3!}}
+ \cdots \\
&= 1 + \sum_i q_i k_i
+ \frac{1}{2!}\sum_{i,j}q_i q_j k_i k_j
+ \frac{1}{3!}\sum_{i,j,k}q_i q_j q_k k_i k_j k_k
+ \cdots
\end{aligned}
$$

But note that:

$$
\sum_i q_i k_i = q\cdot k
$$

$$
\sum_{i,j}q_i q_j k_i k_j = (q\cdot k)^2
$$

$$
\sum_{i,j,k}q_i q_j q_k k_i k_j k_k = (q\cdot k)^3
$$

Therefore:

$$
\Phi(q)\cdot\Phi(k)
= 1 + (q\cdot k) + \frac{(q\cdot k)^2}{2!}
+ \frac{(q\cdot k)^3}{3!}
+ \cdots
= \exp(q\cdot k)
$$

The seemingly simple operation of calculating $\exp(q\cdot k)$ is mathematically equivalent to mapping vectors $q$ and $k$ into an infinite-dimensional feature space and calculating their inner product there. This infinite-dimensional space contains all possible polynomial combinations of the original vectors, including first-, second-, and third-order terms, and so on.

Linear attention was introduced to solve the $O(n^2)$ problem for long sequences. However, directly using this infinite-dimensional feature mapping would itself introduce a feature space tending toward infinite dimension. Applied to a sequence whose length may itself tend toward infinity, this is beyond what a computer can handle.

We must therefore approximate this function. Common approximation methods follow.

1. $\phi(x)=\mathrm{elu}(x)+1$.

   The ELU function outputs $x$ for $x>0$ and $\alpha(\exp(x)-1)$ for $x\le 0$, usually with $\alpha=1$.

   $\mathrm{elu}(x)+1$: for $x>0$, it is approximately $x+1$ (linear growth); for $x\le 0$, it becomes $\exp(x)$ (exponential growth).

   Design motivation: this function behaves similarly to an exponential function when $x\le 0$ while remaining linear for $x>0$, broadly attempting to mimic the shape of the exponential function. It is finite-dimensional, however, with the same output dimension as the input $x$.

2. $\phi(x)=\mathrm{ReLU}(x)^2$.

   This is a simple polynomial function. $\mathrm{ReLU}(x)$ sets negative values to zero and retains positive values, which are then squared.

   It corresponds to a second-order polynomial kernel and is simple to calculate, but its approximation capability is relatively weak.

3. Random-feature-based methods.

   This is a mathematically more rigorous approximation method. It constructs an explicit, finite-dimensional feature mapping $\phi(x)$ through random projections, so that $\phi(q)\cdot\phi(k)$ approximates a target kernel function.

4. Learned kernel functions.

   Instead of designing $\phi$ manually, let the model learn it. A small neural network (such as a one-layer MLP) can serve as $\phi$ and be optimized through gradient descent during training, allowing it to cooperate more effectively with the overall model on the task.

However, none of these kernel functions fits Softmax well enough, which is the core reason why linear attention performs worse than full attention.

## III. Advantages in Autoregression

Linearized attention has another unique advantage: it can easily be rewritten in recurrent form, making it especially suitable for autoregressive inference.

Let:

$$
S_t = \sum_{j=1}^{t}\phi(k_j)^T v_j
= S_{t-1}+\phi(k_t)^T v_t
$$

$$
Z_t = \sum_{j=1}^{t}\phi(k_j)^T
= Z_{t-1}+\phi(k_t)^T
$$

When generating the $t$th token:

1. We already have the accumulated states $S_{t-1}$ and $Z_{t-1}$ for all previous tokens.
2. We calculate the current token's $k_t$ and $v_t$, then update the states:

$$
S_t = S_{t-1}+\phi(k_t)^T v_t,\quad
Z_t = Z_{t-1}+\phi(k_t)^T
$$

3. Then use the current query $q_t$ to calculate the output:

$$
o_t = \frac{\phi(q_t)S_t}{\phi(q_t)Z_t}
$$

This is like an RNN. When generating token $t$, we already have the complete summaries $S_{t-1}$ and $Z_{t-1}$ of the first $t-1$ tokens. When the new token $t$ is produced, we do not need to revisit the entire history. We only add the current token's contribution $\phi(k_t)^T v_t$ to $S_{t-1}$ and $\phi(k_t)^T$ to $Z_{t-1}$ to obtain the new states $S_t$ and $Z_t$. The current query $q_t$ then interacts with the latest states to obtain the output at the current time.

Therefore, we do not need to store the entire key–value history, only fixed-size states $S$ and $Z$. Each step has constant computational cost $O(1)$ with respect to sequence length, greatly improving the efficiency of long-text generation.

## References

- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). ICML 2020.
- Shen, Z., Zhang, M., Zhao, H., Yi, S., & Li, H. (2021). [Efficient Attention: Attention with Linear Complexities](https://arxiv.org/abs/1812.01243). WACV 2021.
