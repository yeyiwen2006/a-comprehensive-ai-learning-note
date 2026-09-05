---
title: "1.1 Classification of Machine Learning"
chapter_title: "Fundamentals of Deep Learning"
section_id: "01-01"
language: en
source_language: zh
source_docx: "第1部分 深度学习/1.深度学习基础理论/1.1 机器学习的分类.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 1.1 Classification of Machine Learning

In artificial intelligence, the word "learning," as in "machine learning," has a specific meaning. It does not refer to rote memorization, but to a model's ability to find patterns in data automatically. More specifically, it can be divided into the following three types.

## I. Supervised Learning: Learning from "Correct Answers"

This is the most common and most readily understood form of learning.

Core idea: we provide the model with a large amount of data whose "answers" have already been marked (a "labeled dataset"), and require its outputs to be as close as possible to these "correct answers."

An everyday analogy: using an answer key. You (the model) work on an exercise (the input data, such as an image), and then immediately turn to the back of the book to check the answer (the label, such as "this is a cat"). The more exercises you complete, the better you can identify a cat's characteristics (such as pointed ears and whiskers). The next time you see a cat image without an answer, you can recognize it too.

Practical applications:

Image classification: you show an AI millions of images labeled "dog," "car," or "face," and it eventually learns to distinguish them.

AlphaFold2: this is a well-known biological model. Scientists provide it with a protein's "sequence" (the exercise) and tell it the corresponding "3D structure" in the Protein Data Bank (PDB) (the correct answer). Eventually, it learns to predict the 3D structures of new sequences.

## II. Unsupervised Learning: Finding Patterns Independently

This approach is more challenging and closer to autonomous human exploration.

Core idea: we provide the model only with a large amount of data "without answers" and let it discover hidden structures or patterns in the data on its own. (This also includes "self-supervised learning," in which the model creates its own "answers.")

An everyday analogy: organizing a jumble of jigsaw pieces. Imagine someone gives you a large box of mixed jigsaw pieces (unlabeled data), but no picture of the completed puzzle to use as a reference. Your (the model's) task is not to assemble a particular "correct" picture, but to discover patterns on your own. For example, you might put all the "blue sky" pieces together and all the "red roof" pieces together. Although you do not know what the final picture will be, you understand the "similarities" among the pieces.

Practical applications (and branches):

Diffusion models: imagine a clear photograph. You (the model) first learn a "noise-adding" process: how to make the photograph completely indistinct step by step. Then you learn the "reverse operation": starting from an indistinct image full of noise, you gradually "denoise" it and ultimately recover a clear image.

Why is this unsupervised? The only thing needed in this process is a large number of clear images; nobody needs to tell the model "this is a cat" or "this is a dog."

Autoregression in large language models: this is "fill in the blank" on a massive scale. The model reads all the text on the Internet (unlabeled data), and its task is to predict the next word. For example, after seeing "The weather today is very __," it knows from learning enormous amounts of text that "good" is the most likely continuation. It creates its own "labels" from the context.

Variational autoencoder (VAE): this resembles a skilled "sketch artist." It first "compresses" a complex image (the input) into a few key features (encoding), and then tries to "reconstruct" the original image using those features (decoding). Its "scoring criteria" have two components: 1. How similar is the reconstructed image to the original? 2. Is the compression sufficiently simple and consistent with a particular pattern (such as a Gaussian distribution)?

Contrastive learning: the core idea is that "like attracts like." It learns a feature space in which similar data points (called "positive pairs") should be close together, while dissimilar data points (called "negative pairs") should be far apart. Different data augmentations (such as random cropping, random color perturbation, Gaussian blur, rotation, and flipping) are applied to the same data point (such as an image) to create two views that look different but have exactly the same semantics, forming a positive pair; different samples are treated as negative pairs. We want the model's predicted similarity to be as large as possible for positive pairs and as small as possible for negative pairs.

## III. Reinforcement Learning: Learning through "Trial and Error"

This approach learns through "rewards" and "penalties." We do not give the model "correct answers," but an "environment" and a "goal" (a reward function). Through repeated "trial and error," the model must learn how to maximize the "total reward" it can obtain.

An everyday analogy: training a pet dog. You (the environment) want your puppy (the model) to learn to "sit." You do not tell it exactly which muscles to move (there are no labels), and initially the puppy may jump about or spin around. But whenever it happens to "sit," you immediately give it a treat (a reward). The puppy is clever and gradually realizes that the action "sit" brings treats. To maximize its "treat" reward, it increases how often it sits. It has learned a policy.

Practical applications:

AlphaGo: through repeated self-play (trial and error) and policy adjustments based on wins and losses (rewards and penalties), it eventually surpassed human Go champions.

Large language model fine-tuning: after self-supervised training, current large models (such as ChatGPT) also undergo reinforcement learning fine-tuning. They generate several answers, which human evaluators (or another AI) then score (reward). The model learns to generate higher-scoring answers that better match human preferences.

## References

- Jumper, J., Evans, R., Pritzel, A., et al. (2021). [Highly Accurate Protein Structure Prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2). *Nature*, 596, 583–589.
- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html). NeurIPS 2020.
- Kingma, D. P., & Welling, M. (2014). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). ICLR 2014.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). [A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html). ICML 2020.
- Silver, D., Huang, A., Maddison, C. J., et al. (2016). [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://doi.org/10.1038/nature16961). *Nature*, 529, 484–489.
- Ouyang, L., Wu, J., Jiang, X., et al. (2022). [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). NeurIPS 2022.
