---
title: "18.3 Model Pruning"
chapter_title: "Model Compression"
section_id: "18-03"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/18.模型压缩/18.3 模型剪枝.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 18.3 Model Pruning

Model pruning removes "unimportant" parameters (weights) or structures from a model. It is divided into two categories:

## I. Unstructured Pruning

Remove individual weights (set them to 0). Because pruning can be adjusted dynamically according to the model weights, the compression ratio is high, but acceleration usually requires specialized hardware or libraries that support sparse computation.

## II. Structured Pruning

Remove entire structural units, such as filters (channels), neurons, or layers. This more easily provides acceleration on general-purpose hardware, but the compression ratio may be slightly lower.

The procedure usually consists of: train a large model -> assess parameter importance (for example, using absolute magnitudes or gradients) -> remove unimportant parameters -> fine-tune the pruned model. Modern methods often use iterative pruning.

## References

- Han, S., Pool, J., Tran, J., & Dally, W. (2015). [Learning both Weights and Connections for Efficient Neural Network](https://papers.nips.cc/paper_files/paper/2015/hash/ae0eb3eed39d2bcef4622b2499a05fe6-Abstract.html). NeurIPS 2015.
- Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). [Pruning Filters for Efficient ConvNets](https://openreview.net/forum?id=rJqFGTslg). ICLR 2017.
- Frankle, J., & Carbin, M. (2019). [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://openreview.net/forum?id=rJl-b3RcF7). ICLR 2019.
