---
title: "25.1 Retrieval-Augmented Generation"
chapter_title: "Context and Memory"
section_id: "25-01"
language: en
source_language: zh
source_docx: "第4部分 大模型智能体/25.上下文与记忆/25.1 检索增强生成.docx"
status: "manually-rebuilt-from-current-docx"
ocr: "all Word images manually transcribed as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 25.1 Retrieval-Augmented Generation

## I. Basic Methods of Retrieval-Augmented Generation (RAG)

Although LLMs have strong general capabilities, their knowledge coverage, timeliness, and precision in locating knowledge when answering remain limited. We therefore let models retrieve information in real time to provide references for their answers.

(I) Standard RAG workflow

First construct a knowledge base: vector storage of unstructured data such as natural-language text. To preserve semantics and facilitate later similarity search, split text into chunks and train an embedding model to convert each chunk into an embedding vector and index it.

During generation, use a query vector to search the knowledge base by similarity, adding the most relevant content to the context (or adding the entire document containing it). For textual knowledge bases, cosine similarity is often used because it focuses on semantic direction without being affected by irrelevant factors such as text length. To avoid missing exact keywords, it is also often combined with traditional keyword retrieval such as BM25, for example, finalScore = (0.7 \* vectorScore) + (0.3 \* textScore).

Original RAG follows "retrieve first, generate second," a sequential architecture: Query → Retrieve → Generate.

(II) Dynamically deciding to invoke search

We can add a `<search>` token to the vocabulary, triggering a search API when generated. Tool-call examples can be added to training data, or RL can teach the model through autonomous exploration to generate `<search>` when needed, using retrieval to assist the task.

1. Architecture types

(1) Conditional architecture: introduce a judger/router that decides whether and how to retrieve based on question difficulty or type, or let the model learn through RL whether to search, that is, whether to generate `<search>`.

(2) Branching architecture: execute multiple paths in parallel and combine results through ensembling or ranking.

Probability ensembling: feed each retrieved document separately to the LLM to calculate generation probabilities, then fuse the resulting distributions with weights.

Candidate ranking: have the LLM generate an answer from each document, rank the answers by confidence or relevance, and select the best.

(3) Loop/iterative architecture: alternate retrieval and generation in a closed loop, Retrieve → Generate, suitable for long-text generation or complex multistep reasoning.

Alternating iteration: generate part of the content → use it to retrieve new information → continue generation from the new information → repeat.

Active triggering: monitor generation probabilities. If the LLM "hesitates" over the current word (low probability), immediately trigger retrieval and use the retrieved information to "correct" the generation direction.

Reflection and self-correction: methods such as Self-RAG train LLMs to output special reflection tokens that judge whether retrieval is needed, whether retrieved content is relevant, and whether generated content is useful.

2. Training methods

(1) Supervised fine-tuning: requires high-quality annotated trajectories, and the nondifferentiability of search makes end-to-end optimization difficult.

(2) Retrieval-integrated RL: using only outcome-based rewards, let the model autonomously learn the alternating "think–retrieve–think again–answer" workflow through algorithms such as PPO or GRPO.

Note that, when calculating policy gradients, trajectory y contains both model-generated tokens and fixed tokens returned by the search engine. Training indiscriminately may make the model try to "optimize" retrieved text or be pulled toward its distribution. "Retrieval" is only an action; what RL rewards teach the model is when to output `<search>`. We therefore mask retrieved-text tokens, excluding the `<search>` token itself, and optimize only the other tokens as the RL policy.

(III) Adaptive RAG

Standard RAG often introduces much irrelevant information alongside the needed material. To avoid accumulating irrelevant context, split retrieved information into smaller blocks, input one block at a time, and call the LLM to answer, repeating until the required answer is obtained. Because of the KV cache, time complexity still grows linearly, as in standard RAG. Provided the needed information is not in the final block, this can theoretically accelerate generation.

Larger blocks are more likely to recall the needed information but also introduce more irrelevant information (distractors). Smaller windows reduce interference but require the model to correctly reject more irrelevant blocks, increasing the risk of mistakenly rejecting useful blocks and accumulating errors.

(IV) Sparse and dense retrieval

Sparse retrieval refers to traditional keyword methods such as BM25. Dense retrieval uses embedding vectors and better exploits semantic information.

In practice, dense and sparse indices are often combined. For example, some versions of claude-mem use Chroma semantic indexing alongside SQLite persistence, filtering, and result completion. Candidate counts, time windows, and ranking logic vary by version, so without a fixed release or commit, one version's default constants should not be treated as a stable interface.

## II. Dense Retrieval

(I) Selecting embedding vectors

1. Chunking

Generally, an embedding vector represents 256–1024 tokens or a paragraph, containing a complete argument or logical passage. Similarities between adjacent sentences can also be calculated; a sharp drop signals a topic change and a suitable boundary.

Why do this? Embedding every word or sentence separately offers fine granularity but will distort meaning through lack of context, making retrieval error-prone. Suppose each sentence in a knowledge base gets one embedding and one sentence is "It crashed." Its vector contains the meaning of "crashed" but loses the subject without context: did a server crash, or the stock market? It will be retrieved in many server- or stock-market-related scenarios, recalling much irrelevant content. Conversely, embedding an entire 5000-character paper in one vector requires either a very high dimension, increasing computation, or mixing the meanings of different passages into a fixed-dimensional vector, making its direction ambiguous. A "document vector" usually actually means a chunk vector.

We also avoid abrupt boundaries and typically use 10%–20% overlap, such as 50 tokens, to prevent a critical sentence from being split between chunks and breaking semantic continuity.

2. Improving chunking

Although mainstream, chunking can lose context. A chunk might say "this approach improves accuracy by 20%" without specifying what "this approach" is. Two improvements are: (1) after retrieving a chunk, provide its entire source document to the LLM; (2) when embedding the chunk, provide a compressed summary of the whole document as reference context.

(II) Storage media for embeddings

Generally, retrieved embedding vectors are stored in CPU memory, possibly after quantization. If the dataset is too large, compressed index vectors can remain in memory while the original text they point to is stored on SSD.

Because embeddings are in CPU memory, using the CPU directly during inference avoids time-consuming I/O blocking for smaller datasets. If large datasets and high concurrency require a GPU, store the quantized, compressed index-vector set in GPU memory, then bring candidates such as the top 1000 from CPU to GPU for precise calculation.

(III) Training embedding models

Because "selecting the most relevant text for context" uses nondifferentiable top-k selection, the embedding model cannot be directly trained end to end with the main LLM and needs separate training.

1. Contrastive learning

Use semantically similar texts as positive samples, such as a paper and its abstract, a forum question and answer, or a news title and body. Use semantically different texts as negatives: nonpositive examples within a batch can serve as negatives, and harder examples can be found, such as passages about apples as fruit versus Apple phones.

Suppose we have a query $q$, a relevant positive document $d^+$, and $N$ irrelevant negative documents $d^-$.

We use information noise-contrastive estimation (InfoNCE) loss, currently the most common formula for training embeddings:

$$
\mathcal{L}=-\log\frac{\exp(\mathrm{sim}(q,d^+)/\tau)}{\sum_{i=0}^{N}\exp(\mathrm{sim}(q,d_i)/\tau)}
$$

Here:

- $\mathrm{sim}(u,v)$ is cosine similarity: $\mathrm{sim}(u,v)=\frac{u^\top v}{\lVert u\rVert\lVert v\rVert}$.
- $d^+$ is the positive sample.
- $d_i$ includes 1 positive and $N$ negative samples.
- $\tau$ is a temperature coefficient controlling sensitivity to hard samples.

2. Distilling the generative model's attention distribution

Literal similarity can be deceptive. Whether embeddings are "similar" should depend on the logic of the content, which emerges in complex generation. For example, query: "New targets for treating Alzheimer's disease." Document A is literally similar: a review titled with "Alzheimer's treatment" but containing only old information. Document B is not literally similar: it discusses "abnormal protein folding and neuronal necrosis" without mentioning "Alzheimer's," but its logic concerns treating the disease by addressing protein folding.

We want downstream-task utility to supervise upstream retrieval ranking. Although the embedding model cannot be optimized end to end, its generated probabilities can imitate the probabilities of tokens in the generator's attention through distillation:

Step 1: obtain the generator's "attention distribution" $P_{\mathrm{Teacher}}$.

Suppose $K$ documents $\{D_1,D_2,\ldots,D_K\}$ are retrieved. As the generator produces answer $Y$, we can inspect cross-attention weights, or self-attention weights in a decoder-only model.

Aggregate the generator's attention weights over all tokens in document $i$, for example, by summing or averaging, to obtain importance score $A_i$. Then normalize:

$$
P_{\mathrm{Gen}}(D_i\mid Q)=\frac{\exp(A_i/\tau)}{\sum_{j=1}^{K}\exp(A_j/\tau)}
$$

This $P_{\mathrm{Gen}}$ is the truth: it represents "how much document $i$ actually contributed to generating the answer."

Step 2: obtain the retriever's "retrieval probabilities" $P_{\mathrm{Student}}$.

The retriever, usually a dual-encoder model, calculates query–document dot-product scores $S_i=E_Q\cdot E_{D_i}$. Apply Softmax normalization:

$$
P_{\mathrm{Ret}}(D_i\mid Q)=\frac{\exp(S_i)}{\sum_{j=1}^{K}\exp(S_j)}
$$

Step 3: calculate the loss and distill.

We want the retriever distribution $P_{\mathrm{Ret}}$ to approximate the generator distribution $P_{\mathrm{Gen}}$ as closely as possible. Kullback–Leibler (KL) divergence is commonly used as the loss function:

$$
\mathrm{Loss}=\mathrm{KL}(P_{\mathrm{Gen}}\Vert P_{\mathrm{Ret}})=\sum_i P_{\mathrm{Gen}}(D_i\mid Q)\log\frac{P_{\mathrm{Gen}}(D_i\mid Q)}{P_{\mathrm{Ret}}(D_i\mid Q)}
$$

Crucially, this loss is differentiable and can update retriever parameters, that is, query-encoder weights, through gradient descent. The retriever learns, "For this question, document B should rank first even though its similarity is only $0.6$."

## III. Reranking and Refinement

(I) Rerankers

1. Full reranking

Sparse and dense retrieval obtain the top-k-scoring documents or chunks, but focus primarily on their overall vector semantics, sacrificing accuracy for speed. Further token-level refinement is therefore needed: use a reranker to rank retrieved top-k documents, such as the first 100. Methods include:

(1) LLM evaluation

(2) Cross-encoder

Input construction: concatenate query $q$ and candidate document $d$ into one sequence with separators:

$$
\mathrm{Input}=[\mathrm{CLS}]\oplus q\oplus[\mathrm{SEP}]\oplus d\oplus[\mathrm{SEP}]
$$

Full self-attention: feed the concatenated sequence to a Transformer. Every query token can attend to every document token, allowing the model to understand complex logical relationships, such as the negation words "not" and "but."

Score output: usually take the $[\mathrm{CLS}]$ output vector, pass it through a fully connected linear layer, and output a scalar score $s\in[0,1]$, typically through Sigmoid.

$$
\mathrm{Score}(q,d)=\sigma\left(W\cdot\mathrm{Transformer}(\mathrm{Input})_{[\mathrm{CLS}]}+b\right)
$$

Here, the query does not apply cross-attention to the document (bi-encoder). Instead, query and document are concatenated and scored with self-attention (cross-encoder). The global attention vector corresponding to the query [CLK] is ultimately taken for maximum precision. (W has dimensions 1\*d\_model and multiplies the [CLS] token vector of dimensions d\_model\*1 to yield a scalar.)

Why choose a cross-encoder?

- **Bi-encoder**:
  - Workflow: query $\to$ vector $V_q$, document $\to$ vector $V_d$.
  - Calculation: $V_q\cdot V_d$.
  - Problem: query and document cannot see each other when their vectors are generated. All semantics must fit into a fixed-dimensional vector, a vector bottleneck. For example, "apple" must implicitly contain both "fruit" and "company" when embedded, making it ambiguous.

"Unable to see each other" means that when generating $V_d$, the model does not know $Q$, and when generating $V_q$, it does not know $D$. They are encoded independently. Suppose document $D$ says, "Apple released a new earnings report, and its stock price rose sharply."

- **Scenario A (cross-encoder, no bottleneck)**:
  - Query: "What fruits taste good?" $\to$ after concatenation, the model sees "fruit" and "stock price" and gives a low score.
  - Query: "Technology company news" $\to$ it sees "technology" and "stock price" and gives a high score.
  - The model dynamically understands semantics through interaction.
- **Scenario B (bi-encoder, with a bottleneck)**:
  - Document $D$ must be compressed into a static vector $V_d$.
  - $V_d$ must express all meanings such as "Apple," "company," "profit," and "technology" using 768 numbers. **The polysemy problem**:
  - If $V_d$ overemphasizes "technology," a query for "apple (fruit)" may still be close in vector space despite a semantic mismatch because both contain embedded information about the word "Apple," causing hallucination or incorrect retrieval.
  - If $V_d$ compromises between fruit and technology, it may lie some distance from both "pure fruit" and "pure technology," so neither retrieves it.

In summary, the vector bottleneck means that, regardless of a sentence's length, richness, or ambiguity, the bi-encoder forces it into a fixed-size vector. Without query-time context, this vector can only be a "vague average," losing fine-grained semantic matching.

With a cross-encoder, the model can concatenate query and document during embedding and use the query to decide which document content enters the embedding. The disadvantage is that the document embedding then depends on the embedding and must be calculated online for every query, rather than embedding the document independently in advance.

2. Chunk reranking

With large datasets, tight latency requirements, or limited computation, token-level processing of every entire paper is unrealistic. Usually, only a few small chunks from each paper enter the reranker; for paper retrieval, the abstract is often added alongside them. Common chunking methods are:

(1) Split by paragraph or fixed token count: each paragraph or fixed number of tokens forms a chunk. This is simplest but can lose context or introduce redundancy. Adding compressed context or using 20% overlap with adjacent chunks partially addresses context loss and preserves continuity.

(2) Split into sentences, then merge: calculate similarity between each sentence's embedding and the next. Above a threshold, merge the next sentence into the same chunk; otherwise, place them in separate chunks. This produces several chunks.

After splitting, calculate similarity between the query and each small chunk's embedding, select the top-k chunks per paper, and send them to the reranker.

(II) Refiners

Retrieved documents often contain irrelevant noise. Feeding them directly to the generator wastes tokens and can cause hallucinations. Refiners aim to improve the "signal-to-noise ratio," so content is commonly refined before entering the context window. Methods include:

1. Extractive refinement: a lightweight scoring model, possibly another cross-encoder, scores each sentence to judge whether it contains key information needed for the answer.

2. Summarization-based refinement: an LLM or small BERT model summarizes the retrieved content.

3. LLMLingua compression: use an LLM to predict a token. The easier the prediction, the lower the token's information content (perplexity), and the more it should be compressed.

$$
\mathrm{Keep}(x_t)=\mathbf{1}\!\left[-\log P\!\left(x_t\mid x_{<t}\right)>\tau\right]
$$

## References

- 温睦宁、林江浩、张伟楠、俞勇. (2025). [*Hands-on Learning of Large-Model Agents*](https://haa.boyuai.com/) (translated title; in Chinese). Posts & Telecom Press. ISBN 978-7-115-68638-1.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html). NeurIPS 2020.
- Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://openreview.net/forum?id=hSyW5go0v8). ICLR 2024.
- Jin, B., Zeng, H., Yue, Z., et al. (2025). [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). arXiv:2503.09516.
- Robertson, S., & Zaragoza, H. (2009). [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). Foundations and Trends in Information Retrieval, 3(4), 333-389.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/). Cambridge University Press.
- thedotmack. (n.d.). [claude-mem GitHub repository](https://github.com/thedotmack/claude-mem). Accessed 2026-09-02.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748). arXiv:1807.03748.
- Izacard, G., & Grave, E. (2021). [Distilling Knowledge from Reader to Retriever for Question Answering](https://arxiv.org/abs/2012.04584). ICLR 2021.
- Nogueira, R., & Cho, K. (2019). [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085). arXiv:1901.04085.
- Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084). EMNLP 2019.
- Xu, F., Shi, W., & Choi, E. (2023). [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408). ICLR 2024.
- Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., & Qiu, L. (2023). [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736). EMNLP 2023.
