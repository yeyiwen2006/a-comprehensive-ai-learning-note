---
title: "19.10 门控注意力（Gated Attention）"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.10 门控注意力（Gated Attention）.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.10 门控注意力（Gated Attention）


> 本文是论文阅读笔记，内容代表对应论文方法或作者理解，不应直接视为领域共识或工程最佳实践。

和前面的几种改进专注提升效率不同，Qwen团队提出的门控注意力（Gated Attention）更多是性能的提升。

## 一、针对的问题

1. 低秩瓶颈

在标准的多头注意力机制（Multi-Head Attention, MHA）中，对于任意一个Head，输出Y是按注意力分数对各token的值矩阵加权：

$$
Y=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中注意力矩阵为：

$$
A=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

也就是说，标准注意力可以写成 $Y=AV$。其中V由多个头拼接而成，对于每个头，输出的矩阵的秩受到矩阵尺寸的制约，只能有 $d_{head}$ 维。在讲DeepSeek的创新时我们提到，各个head的向量是高度相关的，因此实际上仍然分布在一个低秩空间中，会导致V可能丢失了高维数据分布中一些维度的信息。而Softmax之后的Attention层以及输出层都是线性的，V如果没能包含某些维度的信息，就无法通过用非线性操作去“拟合”的方式来恢复这些信息。

2. 不合理的权重分配

以前的许多模型在实际运行时，对于所有token的key都和该query相关度都很低的情形，经常会把大量的注意力分数分配给句子的第一个token（比如起始符），即使这个Token没有任何实际语义。这可能是因为所有token的注意力权重之和强制为1，为了保持数值稳定强行把第一个token作为“垃圾桶”。

## 二、解决方案

在注意力层的输出后面引入门控：

$$
Y_{\mathrm{gated}}=(AV)\odot\sigma(XW_{\mathrm{gate}})
$$

令门控矩阵为：

$$
G=\sigma(XW_{\mathrm{gate}})
$$

则门控注意力可以写成 $Y_{\mathrm{gated}}=(AV)\odot G$。其中 $\odot$ 表示逐元素乘法，$G$ 中的每个元素都由输入 $X$ 经过线性变换和 sigmoid 函数得到。

根据 Schur Product Theorem 的推论，对于秩为 $r_1$ 的矩阵 $A$ 和秩为 $r_2$ 的矩阵 $B$，其逐元素乘积 $A\odot B$ 的秩上界为：

$$
\mathrm{Rank}(A\odot B)\le \mathrm{Rank}(A)\times \mathrm{Rank}(B)
$$

注意这里是逐元素乘法关系，不是普通矩阵乘法。可以看出，秩的上界变成了两个矩阵秩的乘积，大大放宽。还可以从几何上理解：

标准Attention的输出本质上是Value向量的凸组合（convex combination）。由于Softmax之后的注意力权重非负且总和为1，输出向量必须落在所有Value向量构成的凸包（convex hull）内部，因此模型无法生成一个位于这些向量外部的点。

Gated Attention在这个输出后再引入门控矩阵 $G\in(0,1)$。门控对向量的每一个维度进行独立缩放，相当于对凸包内部的点做非均匀的拉伸或压缩：当某些维度的门控值接近0时，这些维度会被抑制；当不同维度的门控强度不同时，输出点就不再只是原始Value向量的单纯凸组合。这样就打破了“凸组合”的几何限制，使得输出空间更丰富。

实验证明，门控矩阵往往是一个稀疏矩阵，这也解决了上面所述不合理的权重分配问题。

## 参考文献

- Qiu, Z., Wang, Z., Zheng, B., et al. (2025). [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708). arXiv:2505.06708.
