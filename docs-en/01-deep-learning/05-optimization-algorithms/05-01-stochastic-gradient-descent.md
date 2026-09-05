---
title: "5.1 Stochastic Gradient Descent (SGD)"
chapter_title: "Optimization Algorithms"
section_id: "05-01"
language: en
source_language: zh
source_docx: "第1部分 深度学习/5.优化算法/5.1 随机梯度下降（SGD）.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 5.1 Stochastic Gradient Descent (SGD)

Stochastic gradient descent (SGD) randomly shuffles samples and performs gradient descent using one sample or one batch at a time.

## I. Problems Faced by SGD during Training

Convergence speed: convergence can be extremely slow on certain landscapes, such as narrow "ravines" or saddle points.

Learning-rate selection: all parameters share one learning rate, making it difficult to accommodate different gradient scales across parameters (features).

Adam addresses both problems by combining two advanced optimization ideas: momentum and RMSProp.

## II. SGD May Match AdamW in Reinforcement Fine-Tuning (2025)

The paper *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models* observes that most parameters actually remain unchanged during reinforcement fine-tuning, which is intrinsically sparse. Essentially, semantic relationships have already been learned during pretraining; reinforcement fine-tuning mainly adjusts their probability distributions to reinforce correct reasoning paths. The local optimization landscape is therefore simple and does not require a complex system such as AdamW. AdamW's built-in momentum and adaptation update parameters even when local gradients are small, producing large changes, many of which are noise. SGD adjusts only parameters with significant gradients, matching the actual needs of reinforcement fine-tuning.

Experiments show that full fine-tuning with SGD may change fewer parameters than LoRA while matching full fine-tuning with AdamW. A relatively large learning rate is needed, around 1e-1.

## References

- Robbins, H., & Monro, S. (1951). [A Stochastic Approximation Method](https://doi.org/10.1214/aoms/1177729586). The Annals of Mathematical Statistics.
- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). [Optimization Methods for Large-Scale Machine Learning](https://epubs.siam.org/doi/10.1137/16M1080173). SIAM Review.
- Mukherjee, S., Yuan, L., Hakkani-Tur, D., & Peng, H. (2025). [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711). NeurIPS 2025.
