---
title: "19.7 Sparse Attention"
chapter_title: "Engineering Optimizations for Attention"
section_id: "19-07"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.7 稀疏注意力（Sparse Attention）.docx"
status: "synced-from-docx"
ocr: "manual reconstruction completed; text and formula screenshots transcribed as Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.7 Sparse Attention

## I. The Core Idea

The core idea of sparse attention is that not every pair of tokens needs to interact. By introducing structured sparsity patterns and calculating only some important elements of the attention matrix, it can reduce the $O(n^2)$ complexity of full attention to $O(n\sqrt{n})$, $O(n\log n)$, or approximately linear complexity.

## II. Representative Examples

Blockwise attention: divide the sequence into blocks and calculate attention only within blocks or between specific blocks.

Sliding-window attention: each token attends only to neighboring tokens within a certain window before and after it. This is common in encoders such as BERT.

Dilated attention: similar to dilated convolution, it introduces skips on top of a sliding window to expand the receptive field.

Global + local attention: designate a small number of tokens (such as `<s>`) to have global attention and attend to all tokens, while other tokens use only local attention.

Applications: models such as Longformer and BigBird use this idea to process extremely long documents.

## III. DeepSeek Sparse Attention (DSA, DeepSeek-V3.2, 2025)

### 1. Core Idea: "Coarse Filtering First, Detailed Computation Second"

DSA aims to quickly identify high-value keys without calculating all query–key dot products.

Inputs include:

- Query-side indexing signals: $q^I_{t,j}$ derived from $c_t^Q$.
- Key-side indexing signals: $k_t^I$.
- Weight or bias terms: $w^I_{t,j}$.

The processing workflow includes:

1. Partial RoPE: to reduce computation, the indexer does not require full-precision positional encoding and applies RoPE only partially to the index vectors.
2. Lightning Indexer: this is a lightweight computational module. It receives coarse-grained query and key index vectors and quickly calculates approximate relevance scores.
3. Top-k selector: based on the scores calculated by the indexer, dynamically select the $k$ key–value blocks that the current query attends to most.

### 2. Inference Workflow

Step 1: Down-projection.

The model does not directly use high-dimensional $h_t$ or the complete $c_t^Q$ for retrieval. Instead, a lightweight linear layer $W^I$ projects the input into a low-dimensional space:

$$
q^I_{t,j} = W^I_Q \cdot c^Q_t
$$

$$
k^I_t = W^I_K \cdot c^{KV}_t
$$

Here, $q^I_{t,j}$ is the query-side index vector, and $k^I_t$ is the key-side index vector. The index-vector dimension $d_{index}$ is much smaller than the normal head dimension $d_{head}$. For example, $d_{head}$ may be 128, while $d_{index}$ may be only 32 or 16, significantly reducing computation during indexing.

Step 2: Blockwise segmentation.

Here, "coarse-grained" refers not only to a low vector dimension but also to time-step granularity. DSA usually does not calculate scores for every historical token individually. Instead, it divides the KV cache into fixed-size blocks, for example, 64 tokens per block.

The indexer calculates a representative key index vector for each block, typically by taking the mean of all $k^I$ in the block or sampling specific positions.

Step 3: Lightweight scoring.

Use dot products to calculate relevance scores between the query index vector and block index vectors:

$$
S_{block} = (q^I_{t,j})^T \cdot k^I_{block}
$$

The phrase "partially apply RoPE" in the figure means that, to further save computation while retaining positional information, the indexer applies rotary positional encoding to only a small subset of vector dimensions or uses a simplified positional encoding.

Step 4: Top-k gating.

Select the $k$ blocks with the highest $S_{block}$ scores. Subsequent genuinely "heavy" core-attention computation loads and computes only these $k$ selected blocks, directly ignoring large amounts of irrelevant information.

This method is faster because the original full-attention computational cost is approximately:

$$
O(L \cdot d_{head})
$$

Whereas the indexer's computational cost is approximately:

$$
O\left(\frac{L}{B} \cdot d_{index}\right)
$$

Here, $B$ is the block size. Because $B > 1$ and $d_{index} \ll d_{head}$, the indexer's overhead is usually much smaller than the complete attention computation.

### 3. Training Method

(1) The training objective of the Lightning Indexer.

Because the top-k operation itself is not differentiable, meaning that the main model's prediction error cannot be backpropagated directly through top-k selection to update the indexer, DeepSeek designs a separate supervision signal for the Lightning Indexer.

The Lightning Indexer is trained to predict attention weights between each token's query and each token's key. The training objective is to imitate the original dense-attention weight distribution, specifically the average weight distribution across all attention heads.

The loss function can be understood as a sum of $n$ KL divergences. For token $i$, its attention scores for token $j$ ($j = 1, 2, \ldots, n$) form the probability distribution $p(j \mid i)$. Calculate the KL divergence between the indexer distribution $p_{Indexer}(j \mid i)$ and dense-attention distribution $p_{Dense}(j \mid i)$, then sum over $i = 1, 2, \ldots, n$:

$$
\mathcal{L}_{indexer}
= \sum_{i=1}^{n} D_{KL}\left(p_{Dense}(\cdot \mid i) \,\|\, p_{Indexer}(\cdot \mid i)\right)
$$

(2) Overall training procedure.

Step 1: train the main model with dense attention.

Step 2: freeze the main model, keep dense attention enabled, and initialize and train the Lightning Indexer to align with the main model's attention distribution.

Step 3: enable sparsification through top-k selection, and train the main model and Lightning Indexer simultaneously using their respective loss functions.

## References

- Qiu, J., Ma, H., Levy, O., Yih, S. W., Wang, S., & Tang, J. (2019). [Blockwise Self-Attention for Long Document Understanding](https://arxiv.org/abs/1911.02972). arXiv:1911.02972.
- Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509). arXiv:1904.10509.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). arXiv:2004.05150.
- Zaheer, M., Guruganesh, G., Dubey, K. A., et al. (2020). [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062). NeurIPS 2020.
- Ding, J., Ma, S., Dong, L., et al. (2023). [LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486). arXiv:2307.02486.
- DeepSeek-AI. (2025). [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556). arXiv:2512.02556.
