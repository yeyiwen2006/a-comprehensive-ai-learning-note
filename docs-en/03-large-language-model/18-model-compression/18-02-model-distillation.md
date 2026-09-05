---
title: "18.2 Model Distillation"
chapter_title: "Model Compression"
section_id: "18-02"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/18.模型压缩/18.2 模型蒸馏.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 18.2 Model Distillation

## I. The Core Idea

In deep learning, we usually train a large, complex model to achieve high accuracy. In actual deployment (such as on mobile phones or IoT devices), however, computational and memory constraints prevent us from using this large model directly. We let a small student model learn the "dark knowledge" contained in a large teacher model by imitating its behavior, thereby approaching the large model's performance as closely as possible while retaining a smaller parameter count. We introduce two concepts below:

Hard target: the teacher tells the student, "The answer is A." This corresponds to the traditional cross-entropy loss function: training focuses on increasing the probability of output A, without concern for the probabilities of outputs B and C.

Soft target: the teacher tells the student, "Choose A, but B also makes some sense, while C is completely wrong." This information containing a probability distribution (such as A=0.7, B=0.2, C=0.1) encodes similarity relationships between classes, that is, the "dark knowledge" within the large model.

Model distillation learns from both types of targets simultaneously. (Learning only from soft targets is not sufficient, as that would reduce the probability of outputting the correct answer.)

## II. Mathematical Formulation

### 1. Softmax with Temperature

In a standard classification task, a neural network outputs logits $z_i$, which are passed through Softmax to obtain probabilities $q_i$. In distillation, we introduce the hyperparameter $T$:

$$
q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- When $T = 1$, this is standard Softmax.
- When $T > 1$, the probability distribution becomes softer. This amplifies the probabilities of incorrect classes that were originally close to 0, revealing the implicit relationships between classes learned by the model (for example, when identifying a "dog," the model assigns a higher probability to resemblance to a "cat" than to resemblance to a "car").

### 2. Loss Function

The student model's total loss function $L$ usually consists of two parts:

$$
L = \alpha L_{KD} + (1 - \alpha)L_{CE}
$$

Here:

- $L_{KD}$ (distillation loss / soft loss): measures the difference between the soft probability distributions produced by the student and teacher models. KL divergence is commonly used. Note that a high temperature $T$ is used in this calculation.
- $L_{CE}$ (student loss / hard loss): measures the difference between the student model's output and the ground-truth labels. Standard cross-entropy loss is commonly used. Note that $T = 1$ is used in this calculation.
- $\alpha$: a hyperparameter that balances the weights of the two losses.

Note that the KL divergence here is forward KL, because:

The standard mathematical definition of KL divergence is:

$$
D_{KL}(P \Vert Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]
$$

Notice the expectation $\mathbb{E}_{x \sim P}$ in the formula. This means that we sample data from distribution $P$ to calculate this integral or sum.

In the context of machine learning, whichever distribution generates the sampled data in the physical world (or computational graph) must appear first in the KL divergence.

### 3. Mathematical Derivation: Why Does High Temperature Work?

Assuming that the logits $z_i$ are small relative to $T$ (that is, $z_i / T \ll 1$), we can apply a Taylor expansion to the Softmax function. The Softmax output $q_i$ is approximately:

$$
q_i \approx \frac{1 + z_i / T}{N + \sum_j z_j / T}
$$

Assuming that the logits have mean 0 (that is, $\sum_j z_j = 0$), this simplifies to:

$$
q_i \approx \frac{1 + z_i / T}{N}
$$

Suppose we use a cross-entropy loss function (or KL divergence) for distillation. Its gradient with respect to the student model's logit $z_i^{(Student)}$ generally takes the form:

$$
\frac{\partial L}{\partial z_i^{(Student)}} = \frac{1}{T}\left(q_i^{(Student)} - q_i^{(Teacher)}\right)
$$

Note: the $1/T$ comes from the coefficient introduced by differentiating $z_i / T$ using the chain rule.

Now substitute the high-temperature probability approximation above for $q_i$:

$$
\begin{aligned}
\frac{\partial L}{\partial z_i^{(Student)}} &\approx \frac{1}{T}\left[\left(\frac{1 + z_i^{(Student)} / T}{N}\right) - \left(\frac{1 + z_i^{(Teacher)} / T}{N}\right)\right] \\
&= \frac{1}{NT^2}\left(z_i^{(Student)} - z_i^{(Teacher)}\right)
\end{aligned}
$$

Ignoring the constant coefficient $\frac{1}{NT^2}$, we find that the gradient is proportional to:

$$
\frac{\partial L}{\partial z_i} \propto z_i^{(Student)} - z_i^{(Teacher)}
$$

Recall the gradient of mean squared error (MSE): if we directly define the loss as the MSE between logits:

$$
L_{MSE} = \frac{1}{2}\left(z^{(Student)} - z^{(Teacher)}\right)^2
$$

Its gradient is exactly:

$$
\frac{\partial L_{MSE}}{\partial z^{(Student)}} = z^{(Student)} - z^{(Teacher)}
$$

This proves that, in the high-temperature limit, minimizing KL divergence (or soft cross-entropy) is equivalent to minimizing the mean squared error between logits.

In practice, of course, we generally choose $T$ between 2 and 20, balancing the hard target's "focus on the maximum" against MSE's "balanced attention to the distribution in every direction." In practical terms:

At normal temperature, Softmax turns the largest logit into a probability close to 1 while compressing the other logits into probabilities close to 0.

Teacher Logits: `[10, 5, 2]` -> Softmax: `[0.99, 0.009, 0.001]`

Teacher Logits: `[10, 8, 8]` -> Softmax: `[0.8, 0.1, 0.1]`

If we look only at probabilities, many subtle differences (such as the difference between 5 and 2) are "flattened," making them difficult for the student model to learn.

Logits, by contrast, directly reflect the model's raw confidence in each class.

The logits `[10, 5, 2]` explicitly tell the student that, although class 1 is the strongest, class 2 is much stronger than class 3 (by 3 units). This information is dark knowledge.

The mathematical formulation of model distillation above is, in fact, an embodiment of the idea of "logit matching." Its results are:

- Direction alignment: distillation requires the logit vector $z_s$ generated by the student and the teacher's logit vector $z_t$ to point in the same direction. This means that the student must learn not only "who comes first" but also the relative ranking of "second, third, ..., Nth."
- Relative magnitude alignment: $z_i - z_j$ represents the relative log-odds gap between classes $i$ and $j$. Logit matching forces the student model to reproduce this relative gap.

For example, if the teacher considers a "cat" 10 times more similar to a "dog" than a "car" is, the student must also consider a "cat" 10 times more similar to a "dog" than a "car" is.

## III. Different Methods of Model Distillation

1. Response-based: use the teacher model's final-layer output. This is the most classical method (as described above).

2. Feature-based: learn not only the output but also force the student model's intermediate feature maps to fit the teacher model's intermediate layers. Additional convolutional layers are usually needed to match the feature-map dimensions.

Loss function:

$$
L_{\mathrm{Feature}} = \left\Vert F_{\mathrm{Teacher}}(x) - W_{\mathrm{reg}}\left(F_{\mathrm{Student}}(x)\right) \right\Vert^2
$$

3. Sample-relation-based:

For the same batch of $N$ samples, obtain their feature vectors in the teacher model's feature space (generally vectors from one layer) and in the student model's feature space. Because the teacher and student feature vectors have different dimensions, we cannot calculate a loss directly. We therefore consider calculating pairwise "similarities" (scalars) between vectors.

We calculate the pairwise cosine similarities or Euclidean distances between these teacher-model feature vectors to obtain an $N \times N$ "similarity matrix," and obtain the same kind of matrix for the student model. If Euclidean distance is used, we also add a normalization factor to eliminate dimensional differences. Our objective is for these two matrices to resemble each other, that is, structural consistency:

$$
L_{\mathrm{Relation}} = \sum_{i=1}^{N} \sum_{j=1}^{N} L_\delta\left(\mathcal{R}_{ij}^{(T)}, \mathcal{R}_{ij}^{(S)}\right)
$$

Here, $L_\delta$ may be Huber loss or $L_2$ loss.

Huber loss is a robust regression loss function that combines the advantages of mean squared error (MSE/L2) and mean absolute error (MAE/L1).

Simply put, its design philosophy is: "Penalize small errors quadratically (for smooth convergence), and large errors linearly (to prevent exploding gradients)."

The Huber loss $L_\delta(y, f(x))$ is defined for prediction error $a = y - f(x)$ as follows:

$$
L_\delta(a) =
\begin{cases}
\frac{1}{2}a^2, & \text{for } |a| \le \delta \\
\delta \cdot \left(|a| - \frac{1}{2}\delta\right), & \text{for } |a| > \delta
\end{cases}
$$

Here:

- $a$ is the residual (the difference between the true and predicted values).
- $\delta$ (delta) is a hyperparameter serving as the "boundary."

When the error is small ($|a| \le \delta$), the loss is a parabola (like MSE).

- Advantage: it is differentiable near 0, and the gradient decreases as the error decreases, helping the model converge finely during the final stage without oscillating near 0 as MAE does.

When the error is large ($|a| > \delta$), it becomes a straight line (like MAE).

- Advantage: its gradient is the constant $\delta$ (or $-\delta$). This means that even extreme outliers do not produce infinitely large gradients, avoiding the MSE situation where "one outlier pulls the parameters of the entire model off course."

Robustness to noise: the teacher model is not perfect, and its relation matrix may contain noise. If the teacher considers the distance between two samples to be 100 (possibly an outlier), while the student considers it to be 1, the loss becomes enormous because MSE grows quadratically ($99^2 \approx 9800$), causing sharp fluctuations in the student model's gradients. With Huber loss, the penalty for this discrepancy is linear and relatively mild.

## IV. On-Policy Distillation

Traditional model distillation is off-policy: the teacher model generates autoregressively, while the student model generates the $i$th token based on the question and tokens $1 \sim i-1$ from the teacher model, making its output distribution approximate the teacher's output distribution. This resembles tracing calligraphy, with the student placing translucent tracing paper over the original. After the teacher has written the first three strokes, the student follows their form to "guess" where the fourth stroke should go. The teacher's handwriting confines the student to a correct path throughout the process.

Because the student model never generates autoregressively on its own during training, when it is deployed in a real environment at test time, it must write from beginning to end without the master's copybook underneath. If it accidentally strays slightly at step 3, then at step 4 it faces an unfamiliar prefix containing an error that it has never encountered in training. Having seen only perfect paths during training, it has no idea how to correct an erroneous path, and subsequent generation collapses completely.

On-policy distillation addresses this problem more effectively: the student model freely explores the environment and autoregressively generates a complete trajectory $y$. This trajectory is retained even if it contains errors, detours, or poor wording. The trajectory, "full of the student's personal style (and even errors)," is fed to the large teacher model. Like a teacher marking homework, the large teacher model provides logits for a standard probability distribution over the entire vocabulary at every time step of this trajectory. The student model then updates its parameters according to this distribution. It explores entirely within the boundaries of its own abilities and state space. What it learns is, "If I reach this step at my current level, how would the large model suggest that I recover or continue?" Because sampling is performed through the student's trajectories, reverse KL is used for learning here:

$$
D_{KL}(\pi_{\mathrm{student}} \Vert \pi_{\mathrm{teacher}}) = \mathbb{E}_{y \sim \pi_{\mathrm{student}}}\left[\log \frac{\pi_{\mathrm{student}}(y|x)}{\pi_{\mathrm{teacher}}(y|x)}\right]
$$

This keeps the model distribution within the reasonable region "delineated" by the teacher, ensuring output quality.

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). arXiv:1503.02531.
- Agarwal, R., Vieillard, N., Zhou, Y., et al. (2024). [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649). arXiv:2306.13649.
- Song, M., & Zheng, M. (2026). [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626). arXiv:2604.00626.
