---
title: "4.8 Sequence-to-Sequence (Seq2Seq) Learning"
chapter_title: "Recurrent Neural Networks"
section_id: "04-08"
language: en
source_language: zh
source_docx: "第1部分 深度学习/4.循环神经网络/4.8 序列到序列（Seq2Seq）学习.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 4.8 Sequence-to-Sequence (Seq2Seq) Learning

Technically, an encoder converts a variable-length input sequence into a fixed-shape context variable $\mathbf{c}$, encoding the input's information in that variable. An RNN can serve as the encoder.

Consider a sample consisting of one sequence (batch size 1). Let the input sequence be $x_1,\ldots,x_T$, where $x_t$ is its $t$th token. At time $t$, the RNN transforms token $x_t$'s input feature vector $\mathbf{x}_t$ and $\mathbf{h}_{t-1}$ (the preceding hidden state) into $\mathbf{h}_t$ (the current hidden state). Let function $f$ describe this recurrent-layer transformation:

$$
\mathbf{h}_t = f(\mathbf{x}_t,\mathbf{h}_{t-1})
$$

In summary, the encoder uses a chosen function $q$ to transform hidden states from all time steps into the context variable:

$$
\mathbf{c} = q(\mathbf{h}_1,\ldots,\mathbf{h}_T)
$$

As noted above, encoder output $\mathbf{c}$ encodes the entire input sequence $x_1,\ldots,x_T$. For a training output sequence $y_1,y_2,\ldots,y_{T'}$, at each time $t'$ (distinct from input/encoder time $t$), the probability of decoder output $y_{t'}$ depends on preceding outputs $y_1,\ldots,y_{t'-1}$ and context $\mathbf{c}$, namely $P(y_{t'} \mid y_1,\ldots,y_{t'-1},\mathbf{c})$.

To model this conditional probability over a sequence, use another RNN as the decoder. At any output time $t'$, it receives the preceding output $y_{t'-1}$ and context $\mathbf{c}$, transforming these with preceding hidden state $\mathbf{s}_{t'-1}$ into current state $\mathbf{s}_{t'}$. Function $g$ can therefore represent the decoder hidden-layer transformation:

$$
\mathbf{s}_{t'} = g(y_{t'-1},\mathbf{c},\mathbf{s}_{t'-1})
$$

After obtaining the decoder hidden state, an output layer and softmax compute the conditional distribution $P(y_{t'} \mid y_1,\ldots,y_{t'-1},\mathbf{c})$ of output $y_{t'}$ at time $t'$.

At every step, the decoder predicts an output-token distribution. As in language modeling, softmax provides the distribution and cross-entropy loss is used for optimization. To load different-length sequences into identically shaped mini-batches, special padding tokens are appended to their ends. Their predictions should be excluded from the loss calculation. The sequence_mask function masks irrelevant entries by setting them to zero, so subsequent computations for irrelevant predictions multiply by zero and yield zero (cross-entropy loss -sigma(pi*logqi), pi=0).

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html). NeurIPS 2014.
