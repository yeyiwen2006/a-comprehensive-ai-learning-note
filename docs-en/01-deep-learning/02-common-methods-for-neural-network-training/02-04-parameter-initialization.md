---
title: "2.4 Parameter Initialization"
chapter_title: "Common Methods for Neural Network Training"
section_id: "02-04"
language: en
source_language: zh
source_docx: "第1部分 深度学习/2.神经网络训练的常用方法/2.4 参数初始化.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 2.4 Parameter Initialization

## I. Xavier (Glorot) Initialization

To avoid vanishing or exploding gradients, we want each layer's output variance to be approximately equal to its input variance. Besides using batch normalization to "pull it back" after each computation, we want the computation itself to preserve this property. In a linear network, such as y_j=sigma(x_i*w_ij)+b_j, the relationship between input and output variance depends on the parameter values, making parameter initialization crucial.

Assume that the weights and inputs are independent and identically distributed and both have mean 0. The weight variance is sigma^2, which we need to determine. The input variance is Var(x_i)=Var(X).

Consider a simple linear layer with output y_j=sigma(x_i*w_ij) (ignoring the bias). For any j, Var(y_j)=sigma(Var(x_i*w_ij))=n_in*Var(x_i*w_ij), where n_in is the number of inputs.

Using the variance formula:

$$
\mathrm{Var}(Z)=E[Z^2]-(E[Z])^2
$$

we have:

$$
E[W_iX_i]=E[W_i]E[X_i]=0\cdot 0=0
$$

$$
\mathrm{Var}(W_iX_i)
=E[(W_iX_i)^2]-0
=E[W_i^2]E[X_i^2]
$$

Since $E[W]=0$:

$$
\mathrm{Var}(W)=E[W^2]-(E[W])^2=E[W^2]
$$

Similarly:

$$
\mathrm{Var}(X)=E[X^2]
$$

Substituting these expressions gives:

$$
\mathrm{Var}(Y)=n_{\mathrm{in}}\cdot E[W_i^2]E[X_i^2]
$$

That is:

$$
\mathrm{Var}(Y)=n_{\mathrm{in}}\cdot \mathrm{Var}(W)\cdot \mathrm{Var}(X)
$$

Forward propagation aims to preserve variance, Var(Y) = Var(X), so Var(W)=1/n_in.

Similarly, during backpropagation, to preserve gradient invariance, we want:

$$
\mathrm{Var}\left(\frac{\partial L}{\partial z^{l-1}}\right)
=\mathrm{Var}\left(\frac{\partial L}{\partial z^l}\right)
$$

Here, z_l is the output of layer l (before activation). The derivation follows:

To derive $\dfrac{1}{n_{\mathrm{out}}}$, we use the assumptions of Xavier (Glorot) initialization: the activation is $\tanh$ or sigmoid, and most inputs $z$ lie in its linear region ($z\approx 0$). In this region, $f(z)\approx z$ and, more importantly, $f'(z)\approx 1$. We also assume that the weights $W$, inputs $a$, and gradients $\dfrac{\partial L}{\partial z}$ all have mean 0:

$$
E[W]=0,\qquad E\left[\frac{\partial L}{\partial z^l}\right]=0
$$

We further assume that $W$, $\dfrac{\partial L}{\partial z^l}$, and $a^{l-1}$ are mutually independent.

Consider how the gradient propagates backward from layer $l$ to layer $l-1$. Let:

$$
z^l=W^la^{l-1}+b^l,\qquad a^l=f(z^l)
$$

Here, $W^l$ has shape $(n_l,n_{l-1})$, $n_{l-1}$ is the number of neurons in layer $l-1$ (fan-in), and $n_l$ is the number of neurons in layer $l$ (fan-out). By the chain rule:

$$
\frac{\partial L}{\partial z^{l-1}}
=\frac{\partial L}{\partial z^l}\cdot
\frac{\partial z^l}{\partial a^{l-1}}\cdot
\frac{\partial a^{l-1}}{\partial z^{l-1}}
$$

In elementwise form:

$$
\frac{\partial L}{\partial z^{l-1}}
=\left((W^l)^\top\cdot\frac{\partial L}{\partial z^l}\right)
\odot f'(z^{l-1})
$$

Consider any neuron $j$ in $\dfrac{\partial L}{\partial z_j^{l-1}}$:

$$
\frac{\partial L}{\partial z_j^{l-1}}
=\left(\sum_{k=1}^{n_l}W^l_{k,j}\frac{\partial L}{\partial z_k^l}\right)\cdot f'(z_j^{l-1})
$$

This sum runs over all neurons $k$ in layer $l$, so $n_l$ is precisely the fan-out of layer $l-1$, denoted $n_{\mathrm{out}}$.

During backpropagation, the gradient of L with respect to a neuron j in the preceding layer (layer l-1) is the sum of the gradients propagated from all neurons in layer l (from 1 to n_out). Thus, Var(W)=Var(w_k,j)=1/n_out is required.

In practice, the harmonic mean of the two values, 2/(n_in+n_out), is used as the variance.

## II. Kaiming Initialization

With a ReLU activation, f'(z) equals 1 for z>0 and 0 for z<0. Its variance:

First use the formula for the variance of a product of independent random variables:

$$
\mathrm{Var}(AB)
=\mathrm{Var}(A)\mathrm{Var}(B)
+\mathrm{Var}(A)E[B]^2
+\mathrm{Var}(B)E[A]^2
$$

Let $A=W^l$ and $B=a^{l-1}$:

$$
\begin{aligned}
\mathrm{Var}(W^la^{l-1})
&=\mathrm{Var}(W^l)\mathrm{Var}(a^{l-1})
+\mathrm{Var}(W^l)(E[a^{l-1}])^2\\
&\quad+\mathrm{Var}(a^{l-1})(E[W^l])^2
\end{aligned}
$$

Applying the assumption $E[W^l]=0$:

$$
\mathrm{Var}(W^la^{l-1})
=\mathrm{Var}(W^l)\left(\mathrm{Var}(a^{l-1})+(E[a^{l-1}])^2\right)
$$

Using the definition of variance, $E[X^2]=\mathrm{Var}(X)+(E[X])^2$, simplifies this to:

$$
\mathrm{Var}(W^la^{l-1})
=\mathrm{Var}(W^l)\cdot E[(a^{l-1})^2]
$$

Substituting into the formula for $\mathrm{Var}(Z^l)$ gives the main forward-propagation equation:

$$
\mathrm{Var}(Z^l)=n_{\mathrm{in}}\cdot \mathrm{Var}(W^l)\cdot E[(a^{l-1})^2]
$$

We next need the relationship between $E[(a^l)^2]$ and $\mathrm{Var}(Z^l)$. Since ReLU satisfies $a^l=\max(0,Z^l)$:

$$
E[(a^l)^2]=E[(\max(0,Z^l))^2]
$$

Assuming $Z^l$ follows a symmetric distribution with mean 0, such as a Gaussian distribution:

$$
\begin{aligned}
E[(a^l)^2]
&=\int_{-\infty}^{\infty}(\max(0,z))^2p(z)\,dz\\
&=\int_{-\infty}^{0}0^2p(z)\,dz+\int_{0}^{\infty}z^2p(z)\,dz\\
&=\int_{0}^{\infty}z^2p(z)\,dz
\end{aligned}
$$

Because $p(z)$ is symmetric, $\int_0^\infty z^2p(z)\,dz$ is exactly half of $\int_{-\infty}^{\infty}z^2p(z)\,dz$. Moreover:

$$
E[(Z^l)^2]=\int_{-\infty}^{\infty}z^2p(z)\,dz
$$

Since $E[Z^l]=0$, $\mathrm{Var}(Z^l)=E[(Z^l)^2]$. Therefore:

$$
E[(a^l)^2]=\frac{1}{2}\mathrm{Var}(Z^l)
$$

Substitute this relationship into the main forward-propagation equation and require stable signals:

$$
E[(a^l)^2]=E[(a^{l-1})^2]
$$

This gives:

$$
E[(a^{l-1})^2]
=\frac{1}{2}\cdot n_{\mathrm{in}}\cdot \mathrm{Var}(W^l)\cdot E[(a^{l-1})^2]
$$

Thus, the variance of W should be 2/n_in.

Backpropagation:

Let sum be the gradient of L with respect to input a of layer l (including the sum of gradients from the n_out neurons in layer l), and let f’(z) be the activation function's gradient. The formula for the variance of a product of independent random variables gives:

$$
\mathrm{Var}\left(\frac{\partial L}{\partial z_j^{l-1}}\right)
=\mathrm{Var}(\mathrm{sum})\mathrm{Var}(f')
+\mathrm{Var}(\mathrm{sum})(E[f'])^2
+\mathrm{Var}(f')(E[\mathrm{sum}])^2
$$

Assuming $E[\mathrm{sum}]=0$:

$$
\mathrm{Var}\left(\frac{\partial L}{\partial z_j^{l-1}}\right)
=\mathrm{Var}(\mathrm{sum})\left(\mathrm{Var}(f')+(E[f'])^2\right)
$$

That is:

$$
\mathrm{Var}\left(\frac{\partial L}{\partial z_j^{l-1}}\right)
=\mathrm{Var}(\mathrm{sum})\cdot E[(f')^2]
$$

We have already derived:

$$
\mathrm{Var}(\mathrm{sum})
=n_{\mathrm{out}}\cdot\mathrm{Var}(W)\cdot
\mathrm{Var}\left(\frac{\partial L}{\partial z}\right)
$$

For ReLU, $E[(f')^2]=0.5$. Substituting:

$$
\mathrm{Var}\left(\frac{\partial L}{\partial z^{l-1}}\right)
=\left(n_{\mathrm{out}}\cdot\mathrm{Var}(W)\cdot
\mathrm{Var}\left(\frac{\partial L}{\partial z}\right)\right)\cdot\frac{1}{2}
$$

Thus, the variance of W should be 2/n_out.

Generally, the forward-propagation variance 2/n_in gives sufficiently good results (initializing with 2/n_in or 2/n_out usually makes little difference).

## References

- Glorot, X., & Bengio, Y. (2010). [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html). AISTATS 2010.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852). ICCV 2015.
