---
title: "15.2 Bahdanau Attention"
chapter_title: "Attention Mechanisms and Transformer"
section_id: "15-02"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/15.注意力机制与Transformer/15.2 Bahdanau注意力.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 15.2 Bahdanau Attention

Bahdanau attention gives the general idea of attention a concrete form to address a central problem in the Seq2Seq models of the time, such as RNN-based encoder–decoder architectures.

In a traditional Seq2Seq model, the encoder must compress the entire input sequence (such as a sentence) into a fixed-length context vector. The decoder then generates the output sequence from this single vector. This creates a clear information bottleneck: for long sentences, a fixed-size vector can hardly retain all the information, degrading model performance and making details near the beginning of the sequence particularly easy to lose.

The core innovation of Bahdanau attention is that, when generating each output word, the decoder can “look back” at the encoder's hidden states at all time steps, dynamically assign attention weights to them for the current generation step, and calculate a context vector specific to that step. This process can be summarized in three steps:

1. Scoring: a small neural network (usually a single-layer feed-forward network) takes the decoder's hidden state from the previous time step as the query (in sentence translation, for example, the query is the part already translated, expressing a “need” to find the next word in the source sentence). Each encoder hidden state (each word in the source sentence to be translated) serves as a key, and the network computes relevance scores (alignment scores) between them.

2. Weighting: the alignment scores are converted by Softmax into weights (an attention distribution). These weights sum to 1 and represent the degree of “attention” paid to different parts of the input sequence when the model generates the current output word.

3. Generating the context vector: a weighted sum of all encoder hidden states is computed, with those states serving as values—that is, word vectors in semantic space—to obtain the context vector for the current time step. This vector focuses on the currently most relevant input information and is then fed to the decoder to help generate the current word.

A key advantage of this method is that alignment (determining correspondences between source- and target-language words) is no longer an independent preprocessing step. Instead, it is learned jointly with translation during end-to-end training.

## References

- Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola. (2023). [*Dive into Deep Learning* (Chinese edition)](https://zh.d2l.ai/). Cambridge University Press.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). ICLR 2015.
