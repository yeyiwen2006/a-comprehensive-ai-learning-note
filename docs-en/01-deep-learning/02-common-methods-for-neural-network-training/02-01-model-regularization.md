---
title: "2.1 Model Regularization"
chapter_title: "Common Methods for Neural Network Training"
section_id: "02-01"
language: en
source_language: zh
source_docx: "第1部分 深度学习/2.神经网络训练的常用方法/2.1 模型正则化.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 2.1 Model Regularization

## I. Mathematical Foundations of Model Regularization

1. Bias and variance

High bias: the mean of the predicted data deviates from the actual mean; underfitting.

High variance: predictions have learned too much noise; overfitting.

2. Bayes' formula

The prior probability p(A) is an initial estimate of an event's probability, such as disease incidence.

p(B|A) is the probability of observing evidence B, such as a positive result, given that A occurs.

p(B) is the sum of the probabilities of observing evidence B in the two cases where A occurs and where it does not.

p(A|B) is the further update to the probability estimate after observing evidence B.

3. Applying Bayes' formula to regularization

(1) Basic quantities

p(parameters): the prior term, an initial estimate of the probability of the event "the parameter values are X." This term can be ignored if there is no regularization term.

p(output|parameters): here, "output" refers to the probability of observing output Y given that "the parameter values are X." When the loss function is minimized, p(output|parameters) is maximized (as obtained by maximum likelihood estimation).

p(output): the average probability of obtaining output Y after trying all parameters; it is generally treated as a constant (drawing completely at random from the list of possible output words).

(2) Basic explanation

p(parameters|output)=p(output|parameters)*p(parameters)/p(output), or equivalently, p(parameters|output) is considered proportional to p(output|parameters)*p(parameters). Our goal is to find the parameters most likely to produce a given output, that is, to maximize p(parameters|output). When designing the loss function, we define this situation as its minimum (for example, loss=-logp(parameters|output)). Over successive rounds of training, gradient descent gradually approaches that parameter combination. Each training step transforms the initial parameter probability distribution into a distribution that accounts for the output, and gradient descent moves the parameters toward the values obtained by maximum likelihood estimation.

(3) The ordinary case

Without a regularization term, we make no prior estimate of p(parameters) [sampling initial weights from a Gaussian distribution enables gradient operations without all-zero weights; it merely sets the optimization starting point and does not represent a prior estimate]. Then p(parameters|output) is proportional to p(output|parameters). In other words, when a model with particular parameters is most likely to produce an output, we regard fitting that model's parameters as the best way to make another model produce the same output.

(4) Overfitting

Training sometimes reaches a parameter combination with a large p(output|parameters), but overfitting has actually occurred and p(output|parameters) is smaller on the test set [note that the Bayesian posterior must be computed only from the training set, not the test set, for the reasons discussed earlier]. To suppress overfitting, we must manually adjust the distribution p(parameters) so that the parameters maximizing p(parameters|output) are influenced both by the training probability of producing the correct output, p(output|parameters), and by the prior distribution p(parameters).

(5) After adding regularization

Adding a regularization term to loss is equivalent to controlling the initial probability distribution function p(parameters), making it nonconstant and thereby incorporating our additional preferences into the estimate. A large p(output|parameters) need not imply a small loss (for example, some regularizers penalize complex models), allowing adjustment of the final parameter distribution.

## II. L2 Regularization (Weight Decay)

This adds a term "lamda*1/2*sum of squared parameters" to the loss function. Here, lamda is the regularization strength; the factor 1/2 ensures that the gradient of the regularization term with respect to parameter vector W is lamda*W, with no additional coefficient.

Probabilistic interpretation: because the loss function is obtained by taking a logarithm, as discussed earlier, p(parameters) is actually the value of a Gaussian function at those parameters. This is equivalent to a Gaussian prior that assigns high probability to parameters near 0.

Convergence behavior: because the derivative is approximately 0 near 0, convergence becomes extremely slow near 0 (the convergence rate approaches 0), so parameters generally do not converge to exactly 0. However, very large parameters are prevented (their penalty is larger than under L1).

Effects on reducing model complexity:

(1) All parameters shrink toward zero, substantially weakening high-degree terms, much like constraining high-degree polynomial coefficients.

(2) It avoids weight cancellation caused by feature collinearity (which would incur a large penalty).

How weight cancellation arises:

1. Indeterminacy of parameter solutions

When features are highly correlated (such as $X_1\approx kX_2$), the model cannot distinguish their independent contributions. For example:

- True relationship: $Y=2X_1+3X_2$.
- If $X_1$ and $X_2$ are strongly correlated (such as $X_2\approx 0.5X_1$), the following solutions may all hold: $Y=4X_1+0\cdot X_2$, $Y=0\cdot X_1+6X_2$, and $Y=5000X_1-4998X_2$.

The last case is weight cancellation: the coefficients of $X_1$ and $X_2$ have opposite signs and extremely large absolute values, so their contributions cancel. Their combined result remains close to the true value, but the individual coefficients have poor interpretability.

2. Instability of gradient updates

During gradient descent optimization, the gradient directions of highly correlated features may conflict. If features $X_1$ and $X_2$ are correlated but differ greatly in magnitude (such as $X_2=100X_1$), a tiny change in weight $w_2$ can cause $w_1$ to oscillate sharply. This may ultimately converge to a cancellation state with negative $w_1$ and positive $w_2$.

Weight cancellation has three types of consequences:

1. Reduced model interpretability. In house-price prediction, for example, if "number of rooms" and "number of rooms squared" are both features, the model may assign a positive weight to the former and a negative weight to the latter, producing the counterintuitive misinterpretation that "adding rooms lowers the house price."

2. Increased variance of parameter estimates. Collinearity inflates weight variance: small perturbations in the samples can cause coefficients to fluctuate sharply, reducing model stability.

3. Distorted feature importance. In linear models, the signs and magnitudes of weights are often used to judge the relationship between a feature and the target. Weight cancellation may make a genuinely positively correlated feature appear negatively correlated in the model.

Other benefits: preventing exploding gradients and avoiding ill-conditioned matrices.

## III. L1 Regularization

Adding "lamda*sum of absolute parameter values" to the loss function is equivalent to using a Laplace prior as the initial estimate of the parameters. Because a linear function has the same slope everywhere, some parameters are driven to zero (with L2, the gradient also approaches 0 as a parameter approaches 0, so further updates toward zero almost stop). The remaining parameters are not as close to 0 as under L2 (which penalizes large deviations more strongly).

## IV. Addressing Matrix Ill-Conditioning

Solve the normal equations by fitting: input matrix (known) A × parameter matrix (unknown) X = output matrix (known) b. From the earlier analysis, this is equivalent to minimizing the objective function ||AX-b||^2.

From the perspective of linear transformations: the parameter matrix consists of multiple original vectors (each column is a vector to be transformed), the parameter matrix represents a linear transformation, and the output matrix represents the transformed versions of the original vectors.

If the columns of input matrix X are highly correlated, the linear transformation represented by this matrix effectively "compresses space together." In that case, the parameter matrix—the solution of the matrix equation, or the original vectors—can change enormously in response to tiny fluctuations in the transformed vectors. Mathematically, regularization is equivalent to addressing ill-conditioning by changing the linear transformation matrix. (Ridge regression is L2 regularization. The derivation of the normal equations from the objective follows; matrix multiplication distributes over addition when dimensions match.)

The objective function of ridge regression is:

$$
J(\mathbf{x})=\lVert A\mathbf{x}-\mathbf{b}\rVert^2+\lambda^2\lVert\mathbf{x}\rVert^2
$$

Here, $A$ is an $m\times n$ design matrix ($m$ samples and $n$ features), $\mathbf{x}$ is the $n\times 1$ parameter vector to be determined, $\mathbf{b}$ is the $m\times 1$ observation vector, and $\lambda>0$ is the regularization parameter.

Expanding the objective in matrix form:

$$
\begin{aligned}
J(\mathbf{x})
&=(A\mathbf{x}-\mathbf{b})^\top(A\mathbf{x}-\mathbf{b})
+\lambda^2\mathbf{x}^\top\mathbf{x}\\
&=\mathbf{x}^\top A^\top A\mathbf{x}
-2\mathbf{b}^\top A\mathbf{x}
+\mathbf{b}^\top\mathbf{b}
+\lambda^2\mathbf{x}^\top\mathbf{x}.
\end{aligned}
$$

Taking the gradient and using the matrix differentiation rules $\nabla_{\mathbf{x}}(\mathbf{x}^\top B\mathbf{x})=2B\mathbf{x}$ (when $B$ is symmetric) and $\nabla_{\mathbf{x}}(\mathbf{c}^\top\mathbf{x})=\mathbf{c}$ gives:

$$
\nabla_{\mathbf{x}}J(\mathbf{x})
=2A^\top A\mathbf{x}-2A^\top\mathbf{b}+2\lambda^2\mathbf{x}
$$

Set the gradient to zero to minimize the objective:

$$
2A^\top A\mathbf{x}-2A^\top\mathbf{b}+2\lambda^2\mathbf{x}=0
$$

Factor out $\mathbf{x}$ and introduce the identity matrix $I$:

$$
(A^\top A+\lambda^2I)\mathbf{x}=A^\top\mathbf{b}
$$

These are the normal equations for ridge regression. Adding $\lambda^2I$ to $A^\top A$ ensures that $A^\top A+\lambda^2I$ is closer to being strictly positive definite, alleviating singularity or ill-conditioning of $A^\top A$. If this matrix is invertible, the unique solution is:

$$
\mathbf{x}=(A^\top A+\lambda^2I)^{-1}A^\top\mathbf{b}
$$

## V. Dropout

1. Basic idea

During training, dropout randomly sets neuron activations to zero with probability $p$ ("dropping" them), forcing the network not to rely on particular neurons or feature combinations and thus improving robustness. Its mathematical expression is:

$$
h'=
\begin{cases}
0, & \text{with probability }p,\\
\dfrac{h}{1-p}, & \text{otherwise}
\end{cases}
$$

The scaling $\dfrac{h}{1-p}$ preserves the expected activation, $E[h']=h$, preventing a mismatch in data scale between training and testing.

2. Mechanisms

- Breaking co-adaptation: neurons cannot rely on fixed combinations and must independently learn useful features, reducing sensitivity to noise.
- Ensemble learning effect: a different subnetwork is trained at each iteration, so the final model is equivalent to an ensemble of subnetworks, improving generalization.

Dropout breaks "co-adaptation" in neural networks by randomly dropping neurons. Co-adaptation means that neurons form fixed combinations with excessive mutual dependence, making the model sensitive to noise in the training data and reducing generalization.

Co-adaptation is essentially a fragile dependency in a neural network. If neurons learning particular features through complex weight combinations depend excessively on other neurons' outputs—for example, neuron A works only when neuron B is active—they form a "symbiotic relationship." This dependency causes overfitting, fragility, and reduced redundancy: the model may rely too heavily on noise or accidental feature combinations in the training data; missing features in test data can break the dependency chain and cause prediction failure; and the model struggles to learn multiple independent feature representations.

By dynamically and randomly dropping neurons, dropout forces the network to avoid fixed dependency paths. Each iteration randomly masks some neurons, and the remaining ones form a new subnetwork. This forces each neuron to be independently useful and less dependent on particular partners, while distributing the same function among multiple groups of neurons. During training, the outputs of retained neurons are also multiplied by $\dfrac{1}{1-p}$; at test time, their original values are used directly. This stabilizes the overall activation strength and prevents signal attenuation due to dropping neurons.

Note: imagine that each feature is handled by one neuron. In a neural network that recognizes animals, the output "tiger" needs to be associated with features such as yellow coloring and stripes, but it must not rely excessively on these superficial features (overfitting them and forming relatively fixed neuron combinations). Nor should a feature be handled by only one neuron (which is risky and inflexible because the value can only be adjusted upward or downward). Such dependence would result in poor generalization and robustness when the training data change slightly (such as the absence of yellow in the example).

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html). Journal of Machine Learning Research.
