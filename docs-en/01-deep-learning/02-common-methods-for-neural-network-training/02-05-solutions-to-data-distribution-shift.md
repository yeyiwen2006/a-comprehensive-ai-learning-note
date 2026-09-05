---
title: "2.5 Solutions to Data Distribution Shift"
chapter_title: "Common Methods for Neural Network Training"
section_id: "02-05"
language: en
source_language: zh
source_docx: "第1部分 深度学习/2.神经网络训练的常用方法/2.5 数据分布偏移的解决方案.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 2.5 Solutions to Data Distribution Shift

## I. Problems Caused by Data Distribution Shift

In machine learning, the objective we want to optimize is $\mathrm{Loss}=\frac{1}{N}\sum L(y,f(x))=\mathbb{E}_{p(x,y)}[L(y,f(x))]$. This represents the average loss over all real data, or the loss weighted by their probabilities (densities). What we can actually optimize is $\mathrm{Loss}_{\mathrm{train}}=\frac{1}{n}\sum L(y,f(x))=\mathbb{E}_{p_{\mathrm{train}}(x,y)}[L(y,f(x))]$, the average loss over sampled data, or the loss weighted by the sample probabilities (densities). We want $\mathrm{Loss}_{\mathrm{train}}$ to be as close to $\mathrm{Loss}$ as possible, so that optimizing $\mathrm{Loss}_{\mathrm{train}}$ optimizes $\mathrm{Loss}$. This means we want the real-data and sample probability distributions to match, so that their probability-weighted (density-weighted) losses match.

When inputs are high-dimensional and have many features (such as images), "probability distribution p" refers by default to a probability density function. This density generally cannot be obtained directly; fitting methods will be introduced later.

If the distributions differ—for example, the samples contain many dark images but very few bright images—dark images receive greater weight in the expectation when optimizing $\mathrm{Loss}_{\mathrm{train}}$. The gradient descent direction is then dominated by "reducing $\mathrm{Loss}_{\mathrm{train}}$ for dark images." Bright images receive insufficient training, reducing the image-recognition model's robustness to lighting.

Such shifts can have disastrous consequences. Suppose we train a loan-applicant default-risk model to predict who will repay or default. The model discovers a relationship between applicants' shoes and default risk (those wearing Oxford shoes repay, whereas those wearing sneakers default). It may then favor lending to all applicants in Oxford shoes and reject everyone in sneakers. Once the model starts making decisions based on footwear, customers understand this and change their behavior. Soon, all applicants wear Oxford shoes, without any corresponding improvement in creditworthiness. Introducing model-based decisions into the environment may break the model. When model outputs are used directly (and uncritically) as the next round's inputs, any small bias can be amplified indefinitely until the system "loses control."

## II. Types of Distribution Shift

1. Covariate shift

The feature distribution p(x) changes, but the rule p(y|x) remains unchanged.

This assumes that "x causes y" (for example, factors x determine a company's share price y).

For example:

- Training set $P_{\mathrm{train}}(\mathbf{x})$: real photographs of cats and dogs.
- Test set $P_{\mathrm{test}}(\mathbf{x})$: cartoons of cats and dogs.
- $\mathbf{x}$ changes: the pixel distributions of "photographs" and "cartoons" are entirely different.
- $P(y\mid \mathbf{x})$ remains unchanged: a "cartoon cat" is still labeled "cat" ($y=1$), and a "cartoon dog" is still labeled "dog" ($y=0$). The classification rule is stable.
- Why it fails: during training, the model may have learned to rely on photograph-specific features, such as fur texture and realistic lighting, which are completely absent from the cartoon test set.

2. Label shift

The probability p(y) of a label y occurring among the sample labels changes, but the features associated with a given label, p(x|y), remain unchanged.

This assumes that "y causes x" (for example, disease y causes symptoms x).

For example:

- Training set (summer): $P(\text{influenza})=1\%$, $P(\text{common cold})=10\%$.
- Test set (winter): $P(\text{influenza})=20\%$, $P(\text{common cold})=40\%$.
- $P(y)$ changes: the label "influenza" itself becomes more common.
- $P(\mathbf{x}\mid y)$ remains unchanged: influenza causes the same symptoms (such as fever and coughing) in summer and winter.

Why this is advantageous: labels $y$ are usually low-dimensional, such as 10 classes, whereas features $\mathbf{x}$ are high-dimensional, such as a million pixels. Correcting a shift in $P(y)$ is much easier than correcting a shift in $P(\mathbf{x})$.

3. Concept shift

The relationship between x and y—the rule P(y|x) itself—changes.

For example:

- Input $\mathbf{x}$: "a sweet carbonated drink."
- The rule $P(y\mid\mathbf{x})$ depends on location:
  - In New York, the rule is $P(y=\text{“soda”}\mid\mathbf{x})\approx 1$.
  - In Chicago, the rule is $P(y=\text{“pop”}\mid\mathbf{x})\approx 1$.
  - In Atlanta, the rule is $P(y=\text{“coke”}\mid\mathbf{x})\approx 1$.

This is the hardest problem. If rules change rapidly, the model must learn from scratch. Fortunately, in practice, concept shift is usually gradual.

## III. Case Studies

1. Medical diagnosis

Problem: a cancer detector.

Data: patients = older men; healthy people = university students (the healthy control group).

Shift type: extreme covariate shift.

Reason for failure: the model never learned "cancer versus healthy." It learned "old versus young." It captures any associated spurious correlations (hormone levels, physical activity, and diet), causing it to fail 100% of the time on genuinely healthy older people (a population it has never seen).

2. Autonomous vehicles and tanks:

A "curb detector": augmenting the training set with "game-engine" data.

Problem: the game engine takes a shortcut and renders every "curb" with the same simple texture.

Shift type: covariate shift.

Reason for failure: the model does not learn to recognize "curb geometry." It learns to recognize "that particular game texture," which is useless in the real world.

A "tank detector":

No tank = forest photographed in the morning (with shadows).

Tank present = forest photographed at noon (without shadows).

Reason for failure: the model learns to recognize "shadows," not "tanks."

3. Nonstationary distributions (slow concept shift)

Advertising model: a model from 2009 does not know what an "iPad" is. The rule P(click | "iPad") came into existence in 2010.

Spam filter: spammers continually evolve their vocabulary and techniques. The rule P(spam | email content) keeps changing in response.

Recommender system: after Christmas, the rule P(interested | "Santa hat") changes from 1 to 0.

4. Summary

Face close-ups: covariate shift. (The training set contains no x in which a "face fills the screen.")

United States/United Kingdom: concept shift. (The y corresponding to "football" changes from "American football" to "soccer.")

Imbalanced data types: label shift. (P(y) is balanced during training but imbalanced in reality.)

## IV. Solutions

1. For covariate shift: importance sampling and weighted Loss

1. The true risk we want to optimize is:

$$
R_{\mathrm{test}}=E_{p_{\mathrm{test}}}[L]
$$

2. But we only have training data:

$$
E_{p_{\mathrm{train}}}[\cdots]
$$

3. By introducing the ratio:

$$
\beta(\mathbf{x})=\frac{P_{\mathrm{test}}(\mathbf{x})}{P_{\mathrm{train}}(\mathbf{x})}
$$

we can rewrite the expectation under $P_{\mathrm{test}}$ as a weighted expectation under $P_{\mathrm{train}}$:

$$
R_{\mathrm{test}}=E_{p_{\mathrm{train}}}[\beta(\mathbf{x})\cdot L]
$$

The importance weight is:

$$
\beta(\mathbf{x})=\frac{P_{\mathrm{test}}(\mathbf{x})}{P_{\mathrm{train}}(\mathbf{x})}
$$

Meaning: if a training sample $\mathbf{x}$ is 10 times more likely to occur in the test set than in the training set, it is very important and should receive 10 times the weight.

We cannot explicitly obtain the probability densities of every sample in the two sets, but we can train a discriminator to fit their ratio directly:

It turns out that we do not need the separate values of $P_{\mathrm{test}}$ and $P_{\mathrm{train}}$, only their ratio $\beta(\mathbf{x})$.

Train a discriminator $f(\mathbf{x})$ to distinguish samples from $P_{\mathrm{test}}$ (label 1) from those from $P_{\mathrm{train}}$ (label 0). To minimize its classification loss, the discriminator must implicitly learn the relative densities of these distributions. If its output is:

$$
f(\mathbf{x})=P(z=1\mid\mathbf{x})
$$

Bayes' formula converts this directly into the desired ratio:

$$
\beta(\mathbf{x})
=\frac{P(\mathbf{x}\mid z=1)}{P(\mathbf{x}\mid z=0)}
\propto\frac{P(z=1\mid\mathbf{x})}{P(z=0\mid\mathbf{x})}
=\frac{f(\mathbf{x})}{1-f(\mathbf{x})}
$$

The discriminator is trained with equal amounts of training-set and test-set data, ensuring p(z=1)=p(z=0).

For example, suppose the main model is trained on a dataset dominated by cartoons while its test set mainly contains real-world images. To improve generalization, a discriminator assigns a high probability of test-set origin to the occasional real-world images in the training set, giving them high weights. This improves real-world generalization during test-set training and prevents a severe shift toward a cartoon-recognition model.

2. For label shift: compute a "confusion matrix" C

Consider an AI diagnostic model in a hospital:

- Classes $y$: three classes, $\{0:\text{healthy},1:\text{common cold},2:\text{influenza}\}$.
- Features $\mathbf{x}$: symptoms such as fever, coughing, and sore throat.
- Label-shift assumption: $P(\mathbf{x}\mid y)$ remains unchanged, meaning that influenza has the same symptoms in summer and winter.

The training set comes from summer:

- Prevalence $P_{\mathrm{train}}(y)$: $\{0:\text{healthy }80\%,1:\text{cold }19\%,2:\text{influenza }1\%\}$.
- The model learns that influenza is extremely rare and is therefore very reluctant to predict it.

The test set is encountered in winter deployment:

- Unknown true prevalence $P_{\mathrm{test}}(y)$: $\{??,??,??\}$.
- We suspect that influenza prevalence will increase substantially.

The model's limitation: because it was trained on $P_{\mathrm{train}}$, it inherently assumes a prior influenza probability of only 1%. In winter, it systematically underestimates influenza incidence and misclassifies many influenza patients as having a common cold.

To address this, divide the dataset (excluding the test set) into training and validation sets. The validation set also comes from summer (P_train), so true labels are available. Run the model on this validation set, tabulate its "error patterns," and normalize the results to obtain confusion matrix C:

The precise definition of $C_{ij}$ is:

$$
C_{ij}=P_{\mathrm{train}}(\hat{y}=i\mid y=j)
$$

Here, $i$ denotes the row (prediction) and $j$ the column (truth). For example, $C_{1,2}$ is the probability that a patient who truly has influenza ($y=2$) is incorrectly predicted to have a common cold ($\hat{y}=1$).

Suppose the computed $C$ is as follows (columns are true labels and rows are predicted labels):

| Predicted \ True | Healthy | Cold | Influenza |
|---|---:|---:|---:|
| Healthy | 0.9 | 0.1 | 0.0 |
| Cold | 0.1 | 0.8 | 0.3 |
| Influenza | 0.0 | 0.1 | 0.7 |

In this matrix, $C_{1,2}=0.3$ means the model has a 30% probability of misclassifying influenza as a cold; $C_{2,2}=0.7$ means it has a 70% probability of correctly recognizing influenza.

Next, the model is deployed in winter (the test set). Running it on, for example, 1,000 new patients gives "contaminated" data:

We do not have these 1,000 patients' true labels $y$, but can observe the predictions $\hat{y}$. Counting the predicted proportions, suppose the model predicts:

- "Healthy" ($\hat{y}=0$): 200 times.
- "Cold" ($\hat{y}=1$): 600 times.
- "Influenza" ($\hat{y}=2$): 200 times.

This gives the average model-output vector:

$$
\mu_{\mathrm{test}}(\hat{y})=[0.2,0.6,0.2]
$$

Note: we do not regard this as the true prevalence. It has been "contaminated" by $C$: for example, the $0.6$ predicted as colds may include many misclassified influenza cases.

We now use the confusion matrix and the law of total probability to recover the true probabilities from the "contaminated" probability data:

In matrix form:

$$
\mu_{\mathrm{test}}(\hat{y})=C\cdot P_{\mathrm{test}}(y)
$$

For colds, the observed average cold-prediction rate in winter, $\mu_{\mathrm{test}}(\hat{y}=1)$, has three components:

1. The probability of being truly healthy ($j=0$) and misclassified as having a cold ($i=1$).
2. The probability of truly having a cold ($j=1$) and correctly classified as having a cold ($i=1$).
3. The probability of truly having influenza ($j=2$) and misclassified as having a cold ($i=1$).

Mathematically:

$$
\begin{aligned}
\mu_{\mathrm{test}}(\hat{y}=1)
&=P(\hat{y}=1\mid y=0)P(y=0)\\
&\quad+P(\hat{y}=1\mid y=1)P(y=1)
+P(\hat{y}=1\mid y=2)P(y=2)\\
&=C_{1,0}\cdot P_{\mathrm{test}}(y=0)
+C_{1,1}\cdot P_{\mathrm{test}}(y=1)
+C_{1,2}\cdot P_{\mathrm{test}}(y=2)
\end{aligned}
$$

This is exactly the dot product of the second row ($i=1$) of matrix $C$ with the true label-distribution vector $P_{\mathrm{test}}(y)$. The result holds for every $i$, so:

$$
\mu_{\mathrm{test}}(\hat{y})=C\cdot P_{\mathrm{test}}(y)
$$

As long as $C$ is invertible, the true label distribution can be recovered from the contaminated prediction distribution:

$$
P_{\mathrm{test}}(y)=C^{-1}\cdot\mu_{\mathrm{test}}(\hat{y})
$$

3. For slow concept shift: online learning

For a nonstationary distribution, for example, do not retrain from scratch. Instead, continually update (fine-tune) the existing model weights with new data so that the model "keeps up" with changes. Bandits are a special case of "online learning" in which the available "actions" (predictions) form a finite set (such as "pull lever 1" or "pull lever 2").

4. Reinforcement learning

This explicitly introduces "rewards." The goal of RL is to learn a policy (how to act) that maximizes long-term reward.

RL is the real solution to the "shoes and loans" problem: the bank (agent) decides to "issue a loan" (action), the environment (customers) responds by "wearing Oxford shoes" (state change), and the bank ultimately observes its "profit" (reward). The RL framework is inherently designed to address this kind of "feedback loop."

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
