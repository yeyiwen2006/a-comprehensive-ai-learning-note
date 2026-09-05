---
title: "19.6 Other KV Compression Methods"
chapter_title: "Engineering Optimizations for Attention"
section_id: "19-06"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.6 KV压缩的其他方法.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.6 Other KV Compression Methods

## I. KV Cache Quantization: An Appropriate Reduction in Precision

A KV cache typically uses FP16 (16-bit) storage by default. The core idea of quantization is to compress the key/value cache from FP16 to INT8, or even INT4, trading lower precision for lower GPU-memory consumption.

Taking INT8 quantization of the key tensor as an example, we can write $K_{int8}=round(K_{fp16}/scale)$. Here, $scale$ is the quantization scale factor, which maps FP16 values into the integer representation range.

The effect is straightforward: compressing FP16 to INT8 approximately halves the KV cache's GPU-memory consumption; further compression to INT4 can reduce it to nearly one quarter of the original. For long-text inference, the larger the KV cache, the more significant the GPU-memory savings from quantization. The specific precision loss depends on the bit width, grouping method, model, and task, and cannot universally be regarded as "almost imperceptible." KIVI is a representative method in this direction: based on differences in the element distributions of keys and values, it applies asymmetric 2-bit quantization per channel and per token, respectively.

## II. K and V Sequence Compression with Information Loss

Some other methods do not process each individual token. Instead, they summarize or cluster the sequence into a small number of prototype "concepts" or "blocks," perform attention over these prototypes, and finally broadcast the results back to the original tokens.

Representative examples:

Pooling/clustering: pool or cluster K and V to reduce their number.

The basic assumption behind such methods is that not all tokens are equally important. Tokens such as the Chinese particles "的" and "了," or "is" and "the," may no longer need to be retained in full after generation.

- H2O (Heavy Hitter Oracle): retain only tokens with relatively high attention scores, that is, heavy hitters, while also retaining recently generated tokens to prevent the model from completely losing its local context.
- KVMerger: identify mergeable sets based on the similarity of key states within the same sequence, then merge the corresponding KV states with Gaussian-kernel weighting to reduce the number of KV entries that must be retained. The method discussed here is KVMerger specifically for the KV cache, not the general-purpose ToMe method for visual tokens.

These methods are currently better suited to mid-to-lower-end applications, especially extremely long-text inference or resource-constrained edge devices. However, they also have clear disadvantages. First, the KV cache becomes dynamically sparse during generation and memory is no longer contiguous, making it difficult to fully exploit GPU parallelism and potentially slowing computation down. Second, in "needle-in-a-haystack" tasks, if a seemingly unimportant token that actually contains critical information, such as a person's name, is pruned early, the model may never be able to find that information later.

Representative techniques include StreamingLLM and H2O. StreamingLLM supports long-text streaming inference by retaining initial attention-sink tokens and recent tokens, whereas H2O emphasizes retaining heavy hitters and recent tokens.

Learned compressors: use a learnable network (such as a low-dimensional projection) to compress K and V. However, this is lossy and irreversible: once the content is compressed into a summary, the original word-for-word and sentence-by-sentence information is lost, making exact quotation difficult.

## References

- Zhang, Z., Sheng, Y., Zhou, T., et al. (2023). [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048). NeurIPS 2023.
- Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453). ICLR 2024.
- Liu, Z., Yuan, J., Jin, H., et al. (2024). [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750). ICML 2024.
- Wang, Z., Jin, B., Yu, Z., & Zhang, M. (2024). [Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks](https://arxiv.org/abs/2407.08454). arXiv:2407.08454.
