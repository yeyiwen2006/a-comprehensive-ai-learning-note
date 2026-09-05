---
title: "4.5 Long Short-Term Memory Networks (LSTMs)"
chapter_title: "Recurrent Neural Networks"
section_id: "04-05"
language: en
source_language: zh
source_docx: "第1部分 深度学习/4.循环神经网络/4.5 长短期记忆网络（LSTM）.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 4.5 Long Short-Term Memory Networks (LSTMs)

## I. Dual-State Architecture

Memory cell ($C_t$): internal, private long-term memory. It resembles a hidden conveyor belt, primarily retaining and transmitting long-term information.

Hidden state ($h_t$): externally exposed short-term memory/output. It is the current time step's output and part of the next step's input.

## II. Core Steps and Gates

Step 1: compute the candidate memory cell:

$$
\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)
$$

Step 2: update the memory cell:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

Here, forget gate $f_t$ multiplies the previous memory state $C_{t-1}$ elementwise. Values near $0$ mean "forget this information," while values near $1$ mean "retain it completely." Input gate $i_t$ determines how much information from each candidate position is written into memory. Together they act somewhat like a GRU's update gate, but their sum is not constrained to $1$, which in fact alleviates vanishing gradients (summing to $1$ entails many fractional weights below one).

Step 3: compute the current hidden state (output):

$$
h_t = o_t \odot \tanh(C_t)
$$

Output gate $o_t$ bridges the memory cell and hidden state, deciding what to read from $C_t$ into $h_t$.

An intuitive interpretation:

An LSTM resembles an office with strict document-management procedures:

Forget gate: decides which old documents in the archive ($C_t$) should be shredded.

Input gate: decides how many new documents received today ($\tilde{C}_t$) should be filed.

Output gate: decides which archived documents to show to a visitor ($h_t$).

Questions to consider:

1. An LSTM has no reset gate like a GRU's, so irrelevant information such as HTML tags affects the candidate memory computation. How does it avoid the resulting problems?

When encountering HTML tags such as `<div>` or `</p>`, a trained LSTM makes $i_t \approx 0$ at the input-gate positions affected by those tags. Meanwhile, forget gate $f_t$ can remain close to $1$, fully preserving the previous memory. Even if this handling is imperfect, the output gate provides an additional "safeguard."

2. Why does it effectively address vanishing gradients in conventional RNNs?

The LSTM memory-update formula is:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

It is a weighted sum of two parts: previous memory $C_{t-1}$ and current candidate memory $\tilde{C}_t$.

During backpropagation, we compute the gradient of $C_t$ with respect to $C_{t-1}$. By differentiation:

$$
\frac{\partial C_t}{\partial C_{t-1}} = f_t
$$

This ignores the candidate memory's dependence on $C_{t-1}$ because we typically consider only the direct dependency. The candidate itself is also computed through $C_{t-1}$, but here we focus on the main path.

An important component of this gradient is forget gate $f_t$. Moreover, the gradient is added onto $C_{t-1}$ rather than passing through the derivative of a compressive nonlinearity such as $\tanh$.

In a conventional RNN, $h_t = \tanh(W [h_{t-1}, x_t] + b)$, gradients repeatedly multiply the derivative of $\tanh$ (less than $1$), causing exponential shrinkage. In an LSTM, the memory-cell gradient is directly $f_t$, and this $f_t$ is obtained through sigmoid, ranging from $0$ to $1$. The key is that the gradient does not repeatedly multiply a number less than $1$, but instead multiplies $f_t$. If $f_t$ is close to $1$, the gradient passes to the preceding time step almost without loss.

The memory gradient can therefore remain relatively stable across time steps during backpropagation instead of vanishing exponentially. This resembles a "highway" along which gradients flow directly, alleviating vanishing gradients.

## III. Computing the Three Gates

Forget gate:

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

Input gate:

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

Output gate:

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

Here, $\sigma$ is sigmoid, keeping all three gates within $(0,1)$.

## IV. Summary

Overall, LSTMs have more parameters and more complex computation than GRUs. They generally perform better on more complex datasets but have no significant advantage over GRUs when data are limited.

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
- Hochreiter, S., & Schmidhuber, J. (1997). [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735). Neural Computation.
