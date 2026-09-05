---
title: "2.3 Batch Normalization (BN)"
chapter_title: "Common Methods for Neural Network Training"
section_id: "02-03"
language: en
source_language: zh
source_docx: "第1部分 深度学习/2.神经网络训练的常用方法/2.3 批量规范化（BN）.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 2.3 Batch Normalization (BN)

## I. Core Idea: Stabilizing Learning

1. The problem: internal covariate shift
This is the central concept proposed by the authors of the BN paper. In simple terms, for a deep network:

During training, the parameters of every layer are constantly updated.

This means that the distribution of the inputs to layer k (the outputs of layer k-1) changes sharply and unpredictably as the preceding layer's parameters are updated. This is internal covariate shift, which causes:

Difficulty in training: subsequent layers must continually adapt to new data distributions, resulting in slow convergence.

The need for extremely careful learning-rate settings: if the learning rate is too high, distributions change too sharply and gradients can explode or vanish; if it is too low, training is excessively slow.

In addition:

- If a single feature has very large numerical values and low variance, the model will rely excessively on it ("taking a shortcut"), causing overfitting.

- BN suppresses this numerical advantage and forces the model to combine multiple features (such as shape and structure) when making judgments, thereby improving generalization.

2. The solution: batch normalization (BN)
BN follows a very intuitive idea: if the problem is excessively sharp distribution changes, then after the input to each layer and before its activation function, force the data distribution back to the form of a standard normal distribution (mean 0 and variance 1), stabilizing it. This also accelerates model training.

## II. Mathematical Principles and Operations

The operations in a BN layer can be divided into the following steps. Suppose we have a mini-batch $B=\{x_1,x_2,\ldots,x_m\}$, where each $x$ can be a sample or the output of a network layer (typically a vector or feature map).

Step 1: compute the mini-batch mean and variance
For the current mini-batch, compute the mean $\mu_B$ and variance $\sigma_B^2$ along each feature dimension.

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

$$
\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

In practice, a tiny constant $\epsilon$ is usually added to the variance to prevent division by zero, giving $\sigma_B^2+\epsilon$.

Step 2: normalize
Use the mean and variance computed above to normalize every sample $x_i$ in the mini-batch into $\hat{x}_i$, with mean 0 and variance 1.

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

Note: a global average is not used here. Besides computational overhead, random fluctuations in mini-batch statistics (distribution differences between batches) resemble injecting noise into the network parameters. This forces the model to learn more robust features and suppresses overfitting. Such implicit regularization is key to BN's improvement in generalization.

Step 3: scale and shift
This is the most important step in a BN layer. Two learnable parameters are introduced: a scaling parameter $\gamma$ and a shifting parameter $\beta$, which adjust the variance and mean, respectively.

$$
y_i=\gamma\hat{x}_i+\beta
$$

Why is step 3 necessary?
Although normalization stabilizes the distribution, it restricts the layer's representational capacity. For example, with a Sigmoid activation, the normalized data are concentrated in its linear region (near 0), making Sigmoid approximately linear and eliminating its nonlinear capability.

Introducing $\gamma$ and $\beta$ gives the BN layer the ability to "restore" the original data distribution.

If the network considers the original distribution optimal, it can learn $\gamma=\sqrt{\sigma^2}$ and $\beta=\mu$, perfectly restoring the prenormalization distribution. It can also learn a distribution with any other mean and variance, letting the network decide which distribution is most beneficial for training.

Combined formula:

$$
\mathrm{BN}(x_i)=\gamma\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta
$$

## III. Differences between BN during Training and Inference

This is a very important detail.

1. Training:

The mean and variance $(\mu_B,\sigma_B^2)$ are computed from the current mini-batch in real time.

At the same time, the network continually computes running averages of the global mean and variance in preparation for inference.

$$
\mu_{\mathrm{global,new}}
= \mathrm{momentum}\cdot\mu_{\mathrm{global,old}}
+ (1-\mathrm{momentum})\cdot\mu_B
$$

$$
\sigma_{\mathrm{global,new}}^2
= \mathrm{momentum}\cdot\sigma_{\mathrm{global,old}}^2
+ (1-\mathrm{momentum})\cdot\sigma_B^2
$$

Here, $\mathrm{momentum}$ is close to $1$, such as $0.9$.

Why use a running average instead of directly averaging globally over all batches?

(1) Mathematically, this gives newer batches greater weight, letting statistics respond quickly to recent distribution changes. If the data distribution changes gradually over time (as in online learning), a running average can dynamically track it and prevent noise from the initially unstable stage from affecting the global estimate.

(2) From an engineering perspective, a global average cannot be computed until all batches have been accumulated, whereas a running average can be computed in real time. In terms of storage and implementation, running averages can be updated within training iterations without an additional procedure. A global average requires additional modules to store, accumulate, and finally compute statistics, increasing code complexity and debugging costs.

2. Inference/testing:

We no longer use mini-batch statistics, because there may be only one sample or the batch statistics may be unstable.

Instead, we normalize directly using the final running-average global mean and variance $(\mu_{\text{global}},\sigma_{\text{global}}^2)$ obtained during training.

$$
y_i=\gamma\frac{x_i-\mu_{\text{global}}}{\sqrt{\sigma_{\text{global}}^2+\epsilon}}+\beta
$$

The benefit is that inference is deterministic, independent of the batch, and highly efficient.

## IV. Placement of BN

Typically, a BN layer is inserted after a fully connected or convolutional layer and before the activation function (such as ReLU).
-> CONV/FC -> BN -> ReLU -> ...

We want to normalize the data entering the activation function first, stabilizing its distribution so that the activation function operates more effectively.

## V. Major Benefits of BN

1. Permitting higher learning rates: in a neural network, the parameter-update step size (learning rate × gradient) is affected by the scale of the input data. If a layer's inputs suddenly become larger, the gradient magnitude surges, potentially producing an excessive update step (explosion) at a fixed learning rate. By stabilizing gradients, BN allows larger learning rates without worrying about exploding or vanishing gradients, substantially accelerating convergence.

2. Reducing strong dependence on initialization: the network becomes considerably less sensitive to weight initialization because BN already ensures a stable data distribution.

3. Acting as a regularizer: the mean and variance differ between mini-batches. This slight noise regularizes model training, reducing overfitting to some extent (dropout can sometimes be reduced or removed).

How does noise "force" the model to become stronger?

(1) Breaking rote memorization (suppressing overfitting)

Problem: the model can easily memorize details of training data (such as noisy pixels in an image) rather than essential features (such as the shape of a cat's ears).

The role of noise: each batch's statistics (mean/variance) act like a slightly distorting filter. An image of the same cat is normalized into "slightly brighter" or "slightly darker" versions in different batches. The model cannot identify a "cat" by absolute pixel brightness and is forced to learn more stable features (such as outlines and textures).

(2) Simulating real-world diversity (improving generalization)

Real-world pattern: cats look different under different lighting and viewing angles, but their essential features remain unchanged.

Simulation through noise: fluctuations in batch statistics are equivalent to automatically generating data variants. For example, batch 1 contains a cat in dim light → the model learns that "cats still have whiskers in the dark." Batch 2 contains a cat in bright light → the model learns "the reflective appearance of cats' eyes in bright light."
(3) Forcing "practice from multiple perspectives" (balancing feature dependence)

Problem: the model may rely excessively on certain features (such as "all cats are yellow").

Intervention through noise: one batch contains only black cats (darker statistics), while another contains only white cats (brighter statistics). The model finds that "color is unreliable" and instead learns more general features such as shape and movement.

4. Alleviating vanishing gradients: controlling the distribution of activation inputs more favorably (avoiding saturated regions) produces larger and more stable gradients.

## VI. Considerations and Variants

Batch-size dependence: BN depends strongly on batch size. If it is too small (such as 1 or 2), estimated means and variances become extremely noisy and can instead harm performance. This motivated later techniques such as Layer Normalization (LN), Instance Normalization (IN), and Group Normalization (GN). They compute statistics along different dimensions and are suitable for small batches or specific tasks (such as NLP and image generation).

Not suitable for every setting: directly applying BN in dynamic networks such as RNNs is difficult. In some generative models, BN may introduce unnecessary within-batch dependencies.

## VII. Summary

Batch normalization (BN) is a revolutionary technique that substantially addresses internal covariate shift in deep-network training by normalizing layer inputs and adding learnable scaling and shifting. Its benefits—faster convergence, higher learning rates, and more stable training—have made it an almost indispensable standard component of modern deep neural networks and directly advanced the design and development of deeper, more powerful models.

It embodies an important idea in deep learning: rather than merely transforming data, let the network learn how to control that transformation to achieve optimal performance.

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
- Ioffe, S., & Szegedy, C. (2015). [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). ICML 2015.
