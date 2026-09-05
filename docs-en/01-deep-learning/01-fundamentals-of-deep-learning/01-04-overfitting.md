---
title: "1.4 Overfitting"
chapter_title: "Fundamentals of Deep Learning"
section_id: "01-04"
language: en
source_language: zh
source_docx: "第1部分 深度学习/1.深度学习基础理论/1.4 过拟合.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 1.4 Overfitting

## I. Overfitting and Underfitting

During training, we can only see the model's Loss on the training set, and we want it to decrease. But the model can "cheat": if it simply "memorizes" all the data, its training Loss is zero, yet it is useless in the real world. A medical model, for example, needs to make judgments about patients it has never seen when it is deployed. It can make effective predictions only if it has genuinely discovered a generalizable pattern. How can we determine whether the model has truly discovered a generalizable pattern rather than simply memorized the data?

1. Training error and generalization error

Training error: the error computed for the model on the training dataset.

Generalization error: the expected model error when the model is applied to infinitely many data samples drawn from the same distribution as the original samples.

The difficulty is that generalization error is an idealized quantity that we can never compute exactly (because we cannot obtain infinite data). In practice, we can only estimate generalization error by applying the model to an independent test set.

2. Overfitting

"Overfitting" refers to the situation in which training error is very low but generalization error is very high.

A familiar example illustrates overfitting: when preparing for China's college entrance examination, achieving good results on mock exams does not guarantee good results on the actual examination. For example, mock exams often simply imitate the previous year's questions. Students who become highly practiced at those question patterns and techniques may nevertheless be at a loss when faced with new questions in the actual examination. This is "overfitting." To handle new examination questions confidently, students must understand and master the transferable, general ideas and methods behind the practice problems. Similarly, a model can make effective predictions only by learning the generalizable patterns behind the data.

Consider also a primary-school pupil who understands the principle behind a formula (such as 1+1=2). Even if the numbers in an examination question change from 1 to 5, the pupil can still calculate the correct answer. This means the pupil has a low "generalization error." Overfitting resembles a student with an exceptional memory who memorizes the answer to every question in the exercise book, earning full marks in routine practice (training error is 0). In the final examination, however, a slight change to a question introduces an unfamiliar number (such as 9), and the student is completely unable to solve it.

3. The relationship between overfitting and model complexity

What constitutes model complexity is itself a complex question. Generally, for neural networks, more parameters, larger ranges of parameter values, and longer training make a model more complex. To explain intuitively how "model complexity" causes overfitting, consider a classic polynomial-fitting example.

Suppose the true pattern in real-world data is a simple quadratic function (a parabola) plus a small amount of random observation error (noise): y=ax^2+bx+c+epsilon. If we fit these data with an excessively complex model, such as a ninth-degree polynomial y = w_9x^9 + w_8x^8 + ... + w_1x + w_0 (with 10 parameters), its 10 degrees of freedom give it enough flexibility to distort itself. It is as though we let the model use an extremely flexible ruler that can bend arbitrarily to connect points on paper: it forces itself through every sample point that has departed from the true trajectory because of measurement error.

To pass through these irregularly distributed noise points, the model is forced to make the coefficients of high-degree terms (such as w_9) very large. For a high-degree term such as x^9, even a tiny change in input x can cause an enormous, explosive change in output y. The fitted curve is no longer a smooth parabola, but an "electrocardiogram" that swings sharply up and down. Although it passes perfectly through every training point, these sharp oscillations produce highly inaccurate predictions at test points (the difference between a ninth-degree function and a quadratic function at randomly chosen points is very likely to be large).

Underfitting is the opposite. For example, when a straight line is used to fit a quadratic function, the model has too little flexibility to capture the patterns in the data, resulting in poor performance.

## II. Increasing the Amount of Training Data

Overfitting means that model complexity is substantially greater than the amount of data. For the same model, fewer samples in the training dataset make overfitting more likely (and more severe); for the same amount of data, greater model complexity makes overfitting more likely. We can therefore increase the amount of training data or reduce model complexity. Here, we discuss increasing the amount of data.

This involves a core assumption: the independent and identically distributed assumption. We assume that training and test data are independently drawn from the same probability distribution. If the data distribution changes (for example, training face recognition on photographs of university students and then using it to recognize older adults in a nursing home), the model will fail.

Even with the same distribution, if there is too little data (for example, only a few thousand images), a complex deep learning model can easily "memorize" every image instead of learning features. Much of the current success of deep learning is due to the emergence of enormous datasets, which allow us to train complex models without excessive overfitting.

In domain-specific tasks, real data are often very scarce or very expensive. For instance, diagnosing a rare cancer from medical images (such as CT or MRI) may involve only a few hundred (or even a few dozen) labeled "positive" images. To address overfitting in this situation, the following approaches can be used:

1. Transfer learning: take a large model pretrained on a large dataset and fine-tune it on these small datasets.

2. Transform existing data: for images, this includes rotation, cropping, scaling, flipping, translation, adding noise, and changing contrast or brightness.

3. Use a variational autoencoder (VAE): during encoding and decoding, the VAE encoder outputs a probability distribution. Rather than reconstructing from a single "point," the decoder reconstructs from a point sampled randomly from this "fuzzy region."

4. Generate data with GANs: we can train a GAN based on a generator–discriminator architecture (such as a variant of DCGAN or WGAN), with the discriminator using the distribution of the few hundred real cancer images as its reference. Once training is complete, the GAN's generator can start from random noise and generate thousands upon thousands of realistic-looking "fake" cancer images that are not identical to the originals.

## III. Model Regularization (Reducing Model Complexity; See the Later Discussion of Model Regularization)

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
