---
title: "2.2 Hyperparameter Optimization and Validation Sets"
chapter_title: "Common Methods for Neural Network Training"
section_id: "02-02"
language: en
source_language: zh
source_docx: "第1部分 深度学习/2.神经网络训练的常用方法/2.2 超参数优化与验证集.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 2.2 Hyperparameter Optimization and Validation Sets

When optimizing parameters on the training set, we cannot treat hyperparameters as optimization variables. They belong to the objective function being optimized: different hyperparameters define different optimization problems.

How, then, do we adjust hyperparameters? We cannot use the test set. If we overfit the training data, we can still detect overfitting by evaluating on test data. But if we overfit the test data, how would we know? We must therefore never rely on test data to select model hyperparameters. However, neither can we rely solely on training data to choose hyperparameters, because we cannot estimate the training data's generalization error.

In practice, the situation is more complicated. Ideally, we would use the test data only once to evaluate the best model or compare several models. In reality, test data are rarely discarded after a single use. We seldom have enough data to use an entirely new test set for every round of experiments.

A common solution is to divide the data into three parts: in addition to training and test datasets, we introduce a validation dataset, also called a validation set. We generally optimize the model (hyperparameters and so forth) based on its validation performance, but must avoid overly aggressive adjustment to prevent overfitting the validation set.

The difficulty is that hyperparameters cannot be optimized with ordinary gradient descent. A direct brute-force search has an enormous search space. Automated optimization would require training a new model for every attempted hyperparameter combination (every step), incurring tremendous computational overhead. We therefore often "tune by hand," using experience and trials on the validation set to adjust hyperparameters.

Furthermore, when training data are scarce, we may not even have enough data for a suitable validation set. A popular solution is K-fold cross-validation. The original training data are divided into K nonoverlapping subsets. Model training and validation are then performed repeatedly: each time, training uses K-1 subsets, and validation uses the remaining subset (the one not used for training in that round). Finally, training and validation errors are estimated by averaging the results of the K experiments.

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
