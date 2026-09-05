---
title: "15.1 Core Principles of Attention"
chapter_title: "Attention Mechanisms and Transformer"
section_id: "15-01"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/15.注意力机制与Transformer/15.1 注意力机制的核心原理.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 15.1 Core Principles of Attention

## I. Problems with Traditional Sequence-Processing Methods and Inspiration from Biological Attention

Before the attention-based Transformer emerged, RNNs were commonly used to process sequential data. An RNN maintains a global context vector called the hidden state $h$. Each time a token $x_t$ is input, $h$ is updated to incorporate its information into $h$ (combining the information already in $h$ with the information brought by $x_t$). The output is ultimately produced from this global context vector. However, this creates two problems. First, the model can only process data sequentially, one item at a time. It handles long-range contextual dependencies poorly (such as a connection between the end and beginning of an article) and is disproportionately affected by nearby noise; introducing gating only slightly alleviates this problem. Second, a single vector can hold too little information to represent the rich relationships in the global context.

To address these problems, researchers began to ask whether the input vectors $x_1,\ldots,x_t,\ldots$ at each time step $t$ could be retained so that their information could be aggregated more flexibly.

In 2017, Google published *Attention Is All You Need*, introducing the Transformer. It was the first mainstream sequence-modeling architecture built entirely on attention, without recurrence or convolution. Attention itself did not originate with the Transformer: in earlier neural machine translation work, Bahdanau and colleagues had already used learnable soft-alignment attention in a recurrent encoder–decoder. The Transformer's key advance was to elevate attention from an auxiliary component to the core of the entire architecture. It adopts “selective focus”: information in the context is retained, and the model autonomously focuses on the parts most relevant at the current moment. For example, suppose a red coffee cup, newspapers, papers, and books are in front of someone. Involuntarily, their visual system tends to be attracted to the red coffee cup. They can also voluntarily and consciously decide to read a book, directing their attention to it.

## II. The Three Core Components of Attention

Attention abstracts this biological process into three core components, which are also essential to understanding how it works:

1. Query

Corresponding example: your subjective intention or task goal of wanting to “read a book.”

Core principle: a query is voluntary and task-driven. It represents “What am I currently looking for?” In a model, the query is usually a vector closely related to the current task. For example, when generating a token, we need not only information from the preceding token but also more context. We therefore use information about this token to look back through the preceding text for additional information.

2. Key

Corresponding example: the prominent features of each object, such as the coffee cup's “red color” or the “textual content” of the newspapers, papers, and books.

Core principle: a key is an involuntary cue from the data itself. It acts as a “label” or “identifier” for each piece of information, to be matched against the query. The system compares the similarity between the query and all keys to decide which information to attend to.

3. Value

Corresponding example: the object itself—the physical coffee cup or the actual text in a book.

Core principle: a value is the sensory input or information ultimately extracted. The purpose of attention is not to select a key, but to select the value associated with that key. The query–key matching results are used to form a weighted aggregation of all values. In a large language model, this amounts to aggregating information from the preceding text.

## III. How Attention Works: An Example

We can now reconstruct the entire process using these three components:

1. Setting the scene (input information): the environment contains five objects (five information sources). Each object has its own key (prominent features such as color and shape) and value (the content of the object itself).

2. Involuntary attention (without an explicit query): there is no strong “subjective intention” (query). The system tends to attend to information with the most prominent keys. For example, the coffee cup with the key “red” is much more visually salient than objects with black-and-white keys. Attention weights therefore naturally favor the coffee cup, amplifying its corresponding value.

3. Voluntary attention (with a query): you form the intention to “read a book”; this is the query.

4. Matching: the system compares the similarity of this query (“want to read a book”) with the keys of all objects in the environment (“news” for the newspaper, “research” for the paper, and “literature” for the book). Weight calculation: the query matches the book's key most closely. The value “book” therefore receives the highest attention weight.

5. Aggregation: using the calculated weights, the system takes a weighted sum of all values (newspaper content, paper content, book content, and so on). Because the book's weight is close to 1 and the other weights are close to 0, the output information comes almost entirely from the value “book.”

Overall, attention is a resource-allocation scheme. It matches a voluntary query against a series of involuntary keys by similarity, computes attention weights, and uses these weights to take a weighted sum of the corresponding values, ultimately producing an output focused on key information.

If $W_Q,W_K,W_V$ are all multiplied by the same input $X$, this is called self-attention.

## IV. Differences from a Fully Connected Layer

A fully connected layer produces its output by “weighting the inputs according to weights.” However, after training, the weights assigned by its $i$-th output to the inputs are fixed. This is relatively inflexible and does not handle situations such as “the third word is important in this sample, but the second word is important in the next sample” well. Attention instead has three weight matrices, $W_Q,W_K,W_V$. Multiplying the input by these different matrices converts it into “what needs to be queried,” “labels used for querying,” and “content retrieved by the query,” respectively. The output is effectively “values weighted by similarity.” Similarities are obtained through transformations such as multiplication involving the input and the weights. Different attention weights can be assigned dynamically to different words according to the characteristics of the current task, allowing the model to focus on key information. Values are also obtained by transforming the input with weights, providing greater flexibility.

By comparison, fully connected or pooling layers treat all input information equally, or process it only according to intrinsic properties of the data (involuntary cues). Attention introduces a dynamic query related to the current task (a voluntary cue), enabling the model to attend dynamically and selectively to different parts of the input according to the context and current goal.

## V. Multi-Head Attention

A single self-attention computation is like understanding a sentence from just one perspective. Relationships between words, however, have multiple dimensions. For example, in “The cat sat on the mat,” the word “sat” relates both to the subject “cat” (who is sitting) and to the location “mat” (where it is sitting). Multi-head attention allows the model to understand contextual relationships from multiple different “perspectives” or “subspaces.”

In traditional attention, one set of queries ($Q$), keys ($K$), and values ($V$) is used to compute attention. Multi-head attention divides the $Q,K,V$ matrices into different parts (mathematically equivalent to projecting their vectors into multiple subspaces of different dimensions), computes attention separately in these subspaces, and finally combines the results by matrix concatenation. This allows the model to capture different types of dependencies.

For example, in language understanding, different heads may learn the following:

Head 1: syntactic dependencies. Focus: subject–verb, verb–object, and modification relationships.

Head 2: semantic roles. Focus: agent, patient, instrument, and location.

Head 3: coreference resolution. Focus: pronouns and their referents.

Head 4: discourse structure. Focus: contrast, causality, and coordination.

Head 5: lexical semantics. Focus: synonyms, antonyms, and hypernym–hyponym relationships.

Head 6: positional patterns. Focus: adjacent words and words at fixed distances.

Head 7: rare patterns. Focus: special structures and idioms.

Head 8: overall balance. Focus: overall consistency checks.

In vision tasks:

Head 1: color features. Head 2: texture features. Head 3: shape features. Head 4: spatial relationships. Head 5: object parts. Head 6: scene context. Head 7: motion patterns. Head 8: saliency detection.

## VI. Mathematical Formulation of a Multi-Head Attention Layer (a Layer, Not a Network)

1. Step 1: computing Q, K, and V

Input vectors: suppose there is an input sequence in which the vector at each time step is $x_i$ (of dimension $d_{\mathrm{model}}$, the input feature dimension of the neural network, such as the dimension of the embedding space in which a word lies).

Projection matrices: $W_Q,W_K,W_V$ (of dimensions $d_{\mathrm{model}}\times d_k$, $d_{\mathrm{model}}\times d_k$, and $d_{\mathrm{model}}\times d_v$, respectively; in multi-head attention with $h$ heads, usually $d_k=d_v=d_{\mathrm{model}}/h$, meaning that multiple heads share them).

Multiplication:

$$
\begin{aligned}
Q_i &= x_i W_Q,\\
K_i &= x_i W_K,\\
V_i &= x_i W_V.
\end{aligned}
$$

Consider the first equation. If it is viewed as a transformation layer, with the embedding vector $x_i$ as the input and the query vector $Q_i$ as the output, then $W_Q$ represents a linear transformation that converts a $d_{\mathrm{model}}$-dimensional input into a $d_k$-dimensional query-vector output through a “fully connected” transformation. From a linear algebra perspective, each row of $W_Q$ represents a basis vector in the query space, and its $j$-th row is multiplied by the $j$-th column of $x_i$.

In practice, for computational efficiency, matrix multiplication is usually used to compute all $i$ simultaneously: $Q=XW_Q$, where $X$ is the input matrix and each row is an $x_i$.

The following example (before splitting into multiple heads) also illustrates the “feature recombination” performed by `W_Q`.

Suppose the input matrix `X` is simplified to three tokens representing “cat,” “chases,” and “mouse,” with a small number of feature entries illustrated as follows:

- `x_1 = [0.8, 0.9, 0.7, ...]`
- `x_2 = [0.6, 0.2, 0.9, ...]`
- `x_3 = [0.7, 0.8, 0.3, ...]`

The first few columns of the projection matrix `W_Q` can be understood as different query features:

- Column 1: subject detection
- Column 2: verb detection
- Column 3: object detection
- `...`

Schematically:

```text
           Col. 1 (subject)   Col. 2 (verb)   Col. 3 (object)   ...
Feature 1       0.9               0.1             0.2          ...
Feature 2       0.8               0.7             0.6          ...
Feature 3       0.1               0.9             0.1          ...
...
```

Then `Q[0,1]` can be approximated as:

$$
Q[0,1] = 0.8 \times 0.1 + 0.9 \times 0.7 + 0.7 \times 0.9 + \cdots = 0.08 + 0.63 + 0.63 + \cdots \approx 1.34
$$

In multi-head attention, converting a higher-dimensional `d_model` input into a lower-dimensional `d_k` output is underpinned by “feature decomposition”: each token's feature vector (the `d_model`-dimensional input) contains many types of features. In multi-head attention, multiplication by `W_Q` first recombines the features, after which different aspects of the features are assigned to different heads. During this process, each head gives different levels of attention to different original features. These differences are reflected in the weights of `W_Q`, but do not mean that the other features have no effect at all. In the subsequent attention computation, each attention head interacts with other tokens around one aspect of the features.

These three computations sometimes also include a bias. Taking $Q$ as an example, $Q=xW_Q+b$, where each element of $b$ corresponds to a column of $W_Q$. Equivalently, the $j$-th column of $Q$ equals the $j$-th column of $xW_Q$ plus the bias $b_j$. For example, $b_1$ denotes the base bias for query feature 1 (such as the baseline tendency toward a “subject query” in the example above), $b_2$ denotes the base bias for query feature 2 (such as the baseline tendency toward a “verb query”), and $b_{512}$ denotes the base bias for query feature 512.

Patterns that the bias vector may learn:

```text
b_Q = [
  0.05,  # Feature 1: inferred from the model's objective, a slight tendency toward a "subject query"
 -0.10,  # Feature 2: slightly suppress a "verb query"
  0.20,  # Feature 3: a stronger tendency toward an "object query"
 -0.05,  # Feature 4: slightly suppress a "modifier query"
  ...
]
```

This means that even when the input is entirely zero, the query vector still has a particular pattern: the model learns “default activation levels” for different query features.

2. Step 2: computing attention weights separately for each head, $\alpha=\mathrm{Softmax}\left(QK_i^T/\sqrt{d_k}\right)$

This step computes the similarity between every query token and every key token, then normalizes it to obtain a similarity matrix. Each attention head computes its own such similarity matrix. The computational complexity is $O(n^2)$.

Suppose we have a query vector $q$ (from $Q$, with shape $[d_k]$, representing a token in a $d_k$-dimensional space) and a key vector $k_i$ (from $K$, also with shape $[d_k]$ and representing a token in a $d_k$-dimensional space). When computing the attention score, the dot product of $q$ and $k_i$ gives a similarity score for the two vectors (usually divided by $\sqrt{d_k}$ for scaling to avoid vanishing gradients).

In practice, of course, neither the query nor the keys consist of just one token. Suppose $Q$ has shape $[n,d_k]$ ($n$ query-token vectors), and $K$ has shape $[m,d_k]$ ($m$ key-token vectors). The result of $QK^T$ is an $[n,m]$ matrix in which each element $(i,j)$ is the dot product of the $i$-th row of $Q$ and the $j$-th row of $K$ (the vector dot product can express the similarity between the $i$-th query token and the $j$-th key token). We then apply Softmax to each row of this matrix (along the key dimension, the second dimension), obtaining the attention-weight matrix $\alpha$ with shape $[n,m]$. For each query token, the normalized matching scores of all key tokens sum to 1.

3. Step 3: multiplying by the values separately for each head, $\mathrm{attn\_output}=\alpha V$

For each attention head, the attention-weight matrix is multiplied by the value matrix. The shape of $\alpha$ is $[n,m]$ ($n$ query-token vectors and $m$ key-token vectors). Its element $(i,j)$ represents how well the $j$-th key token matches the $i$-th query token (the probability of selecting that key token). $V$ has shape $[m,d_v]$ and stores each key token's value in a $d_v$-dimensional space (the dimension of that attention head). Multiplying the $i$-th row of the Output matrix by the $k$-th column of the $V$ matrix means weighting the $k$-th components of all key-token values by their matching scores with the $i$-th query token (the attention that the $i$-th query token allocates to each word), yielding the $k$-th dimension of the new token vector retrieved by the $i$-th query token. Subsequently, token output can be obtained through methods such as Softmax regression using the similarities between this new vector and the other vectors, or through other sampling methods.

$$
o_i = \frac{\sum_{j=1}^{n} \exp\left(\frac{q_i \cdot k_j^T}{\sqrt{d_k}}\right) v_j}{\sum_{j=1}^{n} \exp\left(\frac{q_i \cdot k_j^T}{\sqrt{d_k}}\right)}
$$

In this equation, element $(i,j)$ of $\alpha$ is

$$
\alpha_{i,j} = \frac{\exp\left(\frac{q_i \cdot k_j^T}{\sqrt{d_k}}\right)}{\sum_{t=1}^{n} \exp\left(\frac{q_i \cdot k_t^T}{\sqrt{d_k}}\right)}
$$

Multiplying this by $v_j$ and summing over $j$ gives `o_i`, the `i`-th row vector of `attn_output`.

4. Step 4: output projection after concatenating the heads, $\mathrm{output}=\mathrm{attn\_output}W_O$

In attention, output projection applies a linear transformation to the combined output of multi-head attention. It integrates the lower-level features to obtain higher-level features or other desired results, combining the results of the different heads for decision-making. The shape of `attn_output` is `[seq_len,d_model]`, and $W_O$ has shape $[d_{\mathrm{model}},d_{\mathrm{model}}]$. Its $i$-th column specifies the proportions of the old features used in the new $i$-th feature. The result is a $[\mathrm{seq\_len},d_{\mathrm{model}}]$-dimensional matrix rich in contextual representations, that is, a sequence of the same length as the input. How this is used depends on the task, as discussed later with the full Transformer architecture.

For example:

```text
W_O = [
        New feature 0  New feature 1  New feature 2  New feature 3
  [0.9,    0.1,    0.2,    0.3],    # Old feature 0 (syntactic feature)
  [0.1,    0.8,    0.3,    0.2],    # Old feature 1 (semantic feature)
  [0.2,    0.1,    0.7,    0.4],    # Old feature 2 (positional feature)
  [0.3,    0.2,    0.1,    0.6]     # Old feature 3 (contextual feature)
]
```

Now consider the first token, “cat.” Suppose its four feature values are $x_1,x_2,x_3,x_4$ (the first row of `attn_output`).

New feature 0: [0.9, 0.1, 0.2, 0.3] → primarily depends on the syntactic feature, with slight use of other features.

Purpose: generate a “syntax-enhanced semantic representation,” integrating information about “cat” as the subject, suitable for syntactic analysis tasks.

New feature 0 of the first token (syntax-enhanced representation, position $(0,0)$ in the output matrix): $0.9x_1+0.1x_2+0.2x_3+0.3x_4$.

Column 1: [0.1, 0.8, 0.1, 0.2] → primarily depends on the semantic feature.

Purpose: generate a “pure semantic content representation,” integrating the animal semantics of “cat,” suitable for word-sense understanding tasks.

New feature 1 of the first token (semantic content representation, position $(0,1)$ in the output matrix): $0.1x_1+0.8x_2+0.1x_3+0.2x_4$.

Column 2: [0.2, 0.3, 0.7, 0.1] → primarily depends on the positional feature.

Purpose: generate a “position-aware representation,” integrating the position of “cat” in the sentence, suitable for coreference resolution tasks.

Column 3: [0.3, 0.2, 0.4, 0.6] → uses the different features in a balanced way.

Purpose: generate an “integrated contextual representation,” an integrated representation optimized for the training task, suitable for comprehensive tasks such as translation.

In addition, if the values in one row are significantly larger than those in other rows, this indicates that the model relies more heavily on that old feature (for example, content sentiment analysis relies more on semantic features).

5. Pseudocode example:

```text
# Input: "The cat sat on the mat"
# Translation: "猫 坐在 垫子 上" ("cat / sat / mat / on")
# When generating "猫" ("cat"):

Q = [noun, actor, subject]  # Current decoder state

# Encoder keys:
K = [
    [definite_article, no_lexical_meaning],  # The
    [noun, animal, subject],                # cat
    [verb, past_tense],                     # sat
    [preposition],                         # on
    [definite_article, no_lexical_meaning],  # the
    [noun, object, location]                # mat
]

# Compute similarities:
scores = Q · K^T = [0.1, 0.9, 0.3, 0.1, 0.1, 0.4]  # Most similar to "cat"
weights = softmax(scores / sqrt(d_k)) = [0.05, 0.55, 0.1, 0.05, 0.05, 0.2]

# Take a weighted sum of the values to obtain the output
output = 0.05*V_The + 0.55*V_cat + 0.1*V_sat + ...
# The output mainly contains information about "猫" ("cat"), used to generate the correct translation
```

6. Number of weights

(1) $W_Q,W_K,W_V$: suppose the input dimension is $d_{\mathrm{model}}$, the output uses multi-head attention with $h$ heads, and the dimension of each head is $d_k=d_{\mathrm{model}}/h$ (usually keeping the input and output in embedding spaces of the same dimension). Each head has $3d_{\mathrm{model}}d_k$ weights. With $h$ heads, the total number is $3d_{\mathrm{model}}d_kh=3d_{\mathrm{model}}(d_{\mathrm{model}}/h)h=3d_{\mathrm{model}}^2$. Sometimes the three linear transformations producing $Q,K,V$ also include biases, giving $3d_{\mathrm{model}}^2+3d_{\mathrm{model}}$.

(2) $W_O$: the input dimension is $d_kh$, and the output dimension is $d_{\mathrm{model}}$. The total number of weights is $d_{\mathrm{model}}^2$, or $d_{\mathrm{model}}^2+d_{\mathrm{model}}$ with biases.

Total: $4d_{\mathrm{model}}^2(+4d_{\mathrm{model}})$.

## VII. Code Implementation

```python
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model)  # [embedding dimension, query-token vector dimension]
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, X):
        batch_size, seq_len, d_model = X.shape  # [samples per batch, sequence length, embedding dimension]

        # 1. Q, K, V projections
        Q = self.W_q(X)  # [batch, seq_len, d_model]
        K = self.W_k(X)  # [batch, seq_len, d_model]
        V = self.W_v(X)  # [batch, seq_len, d_model]

        # 2. Split into multiple heads
        # For each sample in the batch, reshape the seq_len*d_model matrix to seq_len*h*d_k,
        # then use transpose(1, 2) to obtain h*seq_len*d_k so it can be split by h.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Current shape: [batch, num_heads, seq_len, d_k]
        # Dimension 1 indexes heads, dimension 2 is sequence position,
        # and dimension 3 contains the different features learned by that head.

        # 3. Compute attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attn_weights, V)

        # 4. Merge heads and restore the original shape
        merged = head_outputs.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # 5. Output projection
        output = self.W_o(merged)
        return output
```

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). ICLR 2015.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
