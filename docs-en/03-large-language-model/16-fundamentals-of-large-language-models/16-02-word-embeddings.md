---
title: "16.2 Word Embeddings"
chapter_title: "Fundamentals of Large Language Models"
section_id: "16-02"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/16.大语言模型的基本原理/16.2 词嵌入（Embedding）.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 16.2 Word Embeddings

## I. The Overall Word-Embedding Pipeline

A tokenizer splits a sentence into tokens, then looks up the integer ID of each token in a vocabulary. The embedding layer receives these integer IDs and converts them into floating-point vectors containing semantic information for subsequent attention layers to process.

## II. Tokenizer

Given a raw text string such as “The cat sat,” the tokenizer first decomposes it into tokens, then uses a word-to-ID mapping dictionary built before training to convert each token into its corresponding integer ID. For example, ["the", "cat", "sat"] becomes [2, 3, 4]. This is not part of what is trained.

## III. Embedding Layer

This is the first layer inside the neural network. It takes the tokenizer's integer-ID sequence, such as [2, 3, 4], and outputs a matrix formed by concatenating the corresponding vectors. Mathematically, this can be represented as a linear matrix computation with no activation function. Its parameters are the embedding vectors of the tokens. For example, a vocabulary of 50,000 words with a 300-dimensional vector per word gives this layer 15,000,000 parameters. During training, the parameters are updated according to inputs and outputs. Correct parameter convergence, with each word embedding effectively expressing its semantics, produces the correct output representation for an input passage. At test time, a forward computation obtains the embedding for each token ID. For example, the 300-dimensional vector for “the” may be [0.12, -0.45, ..., 0.88], for “cat” [0.67, 0.01, ..., -0.23], and for “sat” [-0.11, 0.98, ..., 0.51], producing [[0.12, ...], [0.67, ...], [-0.11, ...]].

For example, the sentence `Don't you love [emoji] Transformers? We sure do.` yields different token and ID sequences with different tokenizers:

- Whitespace tokenization: `["Don't", "you", "love", "[emoji]", "Transformers?", "We", "sure", "do."]`, with corresponding IDs `[1347, 249, 890, 1310, 8219, 568, 909, 791]`.
- spaCy tokenization: `["Do", "n't", "you", "love", "[emoji]", "Transformers", "?", "We", "sure", "do", "."]`, with corresponding IDs `[91, 8123, 21313, 3123, 41251, 151, 9859, 115, 1515, 3134, 4114]`.

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI.
