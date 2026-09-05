---
title: "19.3 Ring Attention"
chapter_title: "Engineering Optimizations for Attention"
section_id: "19-03"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.3 环状注意力（Ring Attention）.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.3 Ring Attention

Ring Attention is a blockwise attention method for distributed multi-GPU/TPU systems. By rotating key/value blocks around a ring of devices, it distributes attention computation for extremely long sequences across devices. Published papers support this algorithmic principle, but without explicit disclosure in a model's technical report, one should not further claim that a particular commercial model uses this implementation.

FlashAttention addresses how to fit extremely long sequences into limited SRAM within a single GPU through tiling; Ring Attention addresses how to fit extremely long sequences (for example, more than 1 million tokens) into a cluster's GPU memory through block rotation across multiple GPUs, breaking through the physical memory limit of a single GPU.

Suppose you want to train a model with a context length of 10 million. Attention requires Q to interact with all K and V. Even with FlashAttention, a single H100 cannot hold such a long KV cache and the intermediate activations. Sequence parallelism is then needed to divide the long sequence across N GPUs.

Ring Attention works as follows: each GPU stores one query block (several rows of the Q matrix, that is, the queries of several tokens), while K and V blocks (each containing several matrix rows) flow among GPUs over a high-speed interconnect like "conveyor-belt sushi." Each GPU processes only the small block passing through it and sends it to the next GPU when finished, never accumulating the data.

This process runs for $P$ rounds among the $P$ nodes in the ring:

1. **Computational Inner Loop**:
   - GPU $i$ uses its local $Q_i$ and the key/value block it currently holds (denoted $K_{\mathrm{curr}}, V_{\mathrm{curr}}$) to calculate attention scores and local outputs.
   - Computation uses FlashAttention's blockwise logic, maintaining local Softmax normalization factors and log-sum-exp statistics.

2. **Communication**:
   - While computing, GPU $i$ sends its current $K_{\mathrm{curr}}, V_{\mathrm{curr}}$ to the next GPU (GPU $i+1$).
   - GPU $i$ simultaneously receives a new key/value block from the previous GPU (GPU $i-1$).

3. **Overlap of Computation and Communication**:
   - The essence of Ring Attention is that matrix multiplication usually takes longer than transmitting key/value blocks, so the two can proceed in parallel.
   - While the system computes block $t$, the network is already transmitting block $t+1$.
   - If computation takes longer than transmission, communication latency is completely hidden, resulting in nearly zero additional communication overhead overall.

The computation depends on the blockwise computation property of Softmax:

Ring Attention works because Softmax can be computed blockwise, which is also the foundation of FlashAttention. Standard Softmax requires global normalization:

$$
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

However, we can split the sequence into two blocks $A$ and $B$, calculate their local statistics separately, and then combine them into global statistics.

First calculate block $A$'s local maximum and local exponential sum:

$$
m_A = \max_{j \in A} x_j,\qquad
l_A = \sum_{j \in A} e^{x_j - m_A}
$$

Then calculate block $B$'s local maximum and local exponential sum:

$$
m_B = \max_{j \in B} x_j,\qquad
l_B = \sum_{j \in B} e^{x_j - m_B}
$$

Finally, combine them using simple mathematical transformations to update the global maximum $m_{\mathrm{global}}$ and global normalization factor $l_{\mathrm{global}}$:

$$
m_{\mathrm{global}} = \max(m_A, m_B)
$$

$$
l_{\mathrm{global}} =
e^{m_A - m_{\mathrm{global}}} l_A
+ e^{m_B - m_{\mathrm{global}}} l_B
$$

Therefore, Ring Attention only needs one $K, V$ block in GPU memory at a time. Once the computation finishes, it updates the statistics and moves that block away.

The detailed computational procedure is as follows:

For GPU $i$, the current round uses only its local query block $Q_i$ and the $K_{\mathrm{curr}}, V_{\mathrm{curr}}$ it currently holds. All the following statistics are calculated separately for each query row; in matrix form, the corresponding maxima and sums are broadcast row-wise.

Calculate local scores:

$$
S = \frac{Q_i K_{\mathrm{curr}}^T}{\sqrt{d}}
$$

Calculate the row maxima of the current block:

$$
m_{\mathrm{block}} = \max(S)
$$

Calculate the local exponential sums of the current block:

$$
l_{\mathrm{block}} = \sum \exp(S - m_{\mathrm{block}})
$$

Calculate the unnormalized attention output:

$$
\tilde O_{\mathrm{block}} = \exp(S - m_{\mathrm{block}}) V_{\mathrm{curr}}
$$

The output O of GPU i is a matrix with the same number of rows as Q_i (if each GPU stores only one query vector, O is a vector). For each query, it is the attention-weighted result over V based on the K and V blocks that have passed through that GPU so far. After each round, update the statistics and output O:

Suppose the previous round maintained $m_{\mathrm{prev}}$, $l_{\mathrm{prev}}$, and $O_{\mathrm{prev}}$, while the current block provides $m_{\mathrm{block}}$, $l_{\mathrm{block}}$, and $\tilde O_{\mathrm{block}}$. Online Softmax merges them as follows.

Update the global maximum:

$$
m_{\mathrm{new}} = \max(m_{\mathrm{prev}}, m_{\mathrm{block}})
$$

Update the normalization factor:

$$
l_{\mathrm{new}} =
e^{m_{\mathrm{prev}} - m_{\mathrm{new}}} l_{\mathrm{prev}}
+ e^{m_{\mathrm{block}} - m_{\mathrm{new}}} l_{\mathrm{block}}
$$

Update the output $O$:

$$
O_{\mathrm{new}} =
\frac{
l_{\mathrm{prev}} e^{m_{\mathrm{prev}} - m_{\mathrm{new}}} O_{\mathrm{prev}}
+ e^{m_{\mathrm{block}} - m_{\mathrm{new}}} \tilde O_{\mathrm{block}}
}{
l_{\mathrm{new}}
}
$$

In this way, each GPU does not need to store the complete $K, V$. By merging local statistics and local outputs in every round, it obtains an attention result equivalent to global Softmax.

## References

- Liu, H., Zaharia, M., & Abbeel, P. (2023). [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889). arXiv:2310.01889.
