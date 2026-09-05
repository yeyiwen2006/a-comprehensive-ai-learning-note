---
title: "4.2 Recurrent Neural Networks (RNNs)"
chapter_title: "Recurrent Neural Networks"
section_id: "04-02"
language: en
source_language: zh
source_docx: "第1部分 深度学习/4.循环神经网络/4.2 循环神经网络（RNN）.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 4.2 Recurrent Neural Networks (RNNs)

## I. Core Idea

The core idea of an RNN is to introduce "memory" or "state," letting the network retain information from previous inputs and use it to influence the current output.

Feedforward networks, such as CNNs and MLPs, assume inputs (and outputs) are mutually independent. Processing one sample does not account for preceding or subsequent samples; there is no concept of "context." The amount of input data is fixed for each forward computation. An $n$-gram model based on this design, with fixed $n$, predicts the next word $x_t$ by looking only at the immediately preceding $n-1$ words $x_{t-1},\ldots,x_{t-n+1}$. But as $n$ grows, it must record probabilities for every possible combination of $n$ words. With a vocabulary of 100,000 words, an $n=4$ model requires $100{,}000^4$ probability values—an astronomical number, infeasible in both computation and storage.

To address this parameter explosion as $n$ increases, we introduce a latent-variable model. Instead of depending directly on every historical word, it relies on an intermediate variable summarizing all historical information, called the hidden state. The new model is:

$$
P(x_t \mid x_{t-1},\ldots,x_1) \approx P(x_t \mid h_{t-1})
$$

The hidden state $h_{t-1}$ is a vector encapsulating all useful information from the sequence's beginning through time $t-1$. Its dimensionality is fixed (such as 256) and does not grow with sequence length, avoiding exponential parameter growth with history length. RNNs based on this design specifically handle sequential data: successive, interrelated input (or output) data points. Examples include time series (stock prices, weather, and sensor readings), natural language (sentences and paragraphs, essentially word sequences), and audio/video (temporally ordered frames or samples).

## II. Mathematical Expression

1. The new hidden layer:

$$
h_t=f(x_t,h_{t-1})=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$

$W_{xh}x_t$: a linear transformation projecting the current input $x_t$ (a numerical vector representing a word) into the hidden space.

$W_{hh}h_{t-1}$: another linear transformation, projecting the previous hidden state (memory) into the hidden space. Weight matrix $W_{hh}$ is the RNN's most important component, determining how past memory influences the current state.

$b_h$: the bias term.

$\phi$: a nonlinear activation such as $\tanh$ or ReLU. The $\tanh$ function compresses values into $(-1,1)$, helping stabilize network outputs.

2. Producing outputs: the new hidden state $h_t$ can usually serve directly as the output or pass through an output layer. The final output at the current time is:

$$
o_t=W_{ho}h_t+b_o
$$

3. Parameter updates: backpropagation through time (BPTT)

(1) Unroll the network over time into a deep feedforward network

Imagine an RNN processing a sequence of length 5, $[x_1,x_2,x_3,x_4,x_5]$. Forward propagation computes:

$$
h_1=f(x_1,h_0)\to h_2=f(x_2,h_1)\to h_3=f(x_3,h_2)\to h_4=f(x_4,h_3)\to h_5=f(x_5,h_4)
$$

From BPTT's perspective, this is a special 5-layer DNN:

Layer 1: input $[x_1,h_0]$, output $h_1$.

Layer 2: input $[x_2,h_1]$, output $h_2$.

...

The special property of this "deep network" is that all "layers" share the same parameters $(W_{xh}, W_{hh}, b_h)$. Layer $t$ receives both the original input $x_t$ and the preceding layer's output $h_{t-1}$.

(2) Backpropagation

Consider a simple RNN with hidden-state update:

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

The final output and loss are:

$$
o_t = W_{hq}h_t + b_q
$$

$$
L = \sum_{t=1}^{T} L_t = \sum_{t=1}^{T} \mathrm{Loss}(o_t, y_t)
$$

A. Gradients of the final output-layer parameters ($W_{hq}, b_q$)

This is the simplest part: $o_t$ depends only on $h_t$, so gradients do not propagate across time steps. They can be computed independently at each step and summed.

$$
\frac{\partial L}{\partial W_{hq}}
= \sum_{t=1}^{T} \frac{\partial L_t}{\partial o_t}\frac{\partial o_t}{\partial W_{hq}}
= \sum_{t=1}^{T}(o_t - y_t) \cdot h_t^T
$$

$$
\frac{\partial L}{\partial b_q}
= \sum_{t=1}^{T} \frac{\partial L_t}{\partial o_t}\frac{\partial o_t}{\partial b_q}
= \sum_{t=1}^{T}(o_t - y_t)
$$

B. Gradients of the recurrent-layer parameters ($W_{xh}, W_{hh}, b_h$)

The total loss $L$ depends on $h_t$, and $h_t$ in turn depends on $h_{t-1}$ and the parameters. This dependency extends back to the sequence's beginning, so computing gradients involves the chain rule across time steps.

Define a key variable: $\delta_t$, the gradient of the loss with respect to the hidden state at time $t$.

$$
\delta_t = \frac{\partial L}{\partial h_t}
$$

1. Initialize at the final time step $T$:

$\delta_T$ comprises two parts: loss from the current output and loss from the future. In many simple cases, however, if $h_T$ affects only $o_T$:

$$
\delta_T = \frac{\partial L_T}{\partial h_T}
= \frac{\partial L_T}{\partial o_T}\frac{\partial o_T}{\partial h_T}
= (o_T - y_T) W_{hq}^T
$$

2. For $t = T-1, T-2, \ldots, 1$, recursively propagate gradients backward:

$$
\delta_t = \frac{\partial L_t}{\partial h_t} + \frac{\partial h_{t+1}}{\partial h_t}\delta_{t+1}
$$

The first term comes from the current output; the second comes from future time steps.

## III. Problems with Conventional RNNs (Motivating GRUs and LSTMs)

1. Vanishing and exploding gradients

This is the best-known RNN problem. Backpropagation multiplies gradients all the way backward across time steps.

If the eigenvalues of $W_{hh}$ (the hidden-state weight matrix) are less than 1, repeated multiplication makes gradients extremely small. Early-step parameters cannot be updated effectively, and errors made early in the sequence are difficult to correct (the network "forgets" distant events). Forcing the model to remember early information would require early parameters to produce a huge gradient that does not decay during backpropagation. This is either mathematically difficult to achieve naturally or causes another problem—exploding gradients that disrupt training of other parameters—leading to training failure.

Conversely, eigenvalues greater than 1 cause exponential gradient growth (exploding gradients), unstable training, drastic parameter updates, and failure to converge.

2. Long-range dependencies

A direct consequence of vanishing gradients is difficulty learning dependencies between distant sequence elements. For example, in "The clouds which have been covering the sky all day are finally starting to ___," predicting the final word "clear" or "rain" requires remembering the much earlier subject "clouds."

The conventional RNN's difficulty: vanishing gradients make it hard to pass information (error) from the first time step all the way to the last. Learning to remember it requires assigning a huge gradient to the first step, which is difficult in practice.

Required mechanism: a "memory cell" that can stably retain key information for a long time, like a computer hard drive, without interference from subsequent operations (gradient calculations).

3. Difficulty ignoring irrelevant information

Example: in webpage sentiment analysis, HTML tags such as `<div>` and `<br>` are unrelated to sentiment and constitute noise.

The conventional RNN's difficulty: it processes every input indiscriminately, so irrelevant tokens also update the hidden state and contaminate useful subsequent information.

Required mechanism: "forgetting" or "skipping," actively selecting which important information should enter and which unimportant information should be ignored.

4. Difficulty resetting the internal state

Example: clear boundaries or topic changes separate different parts of a sequence, such as the beginning of a new book chapter or a market changing from bearish to bullish.

The conventional RNN's difficulty: its hidden state continually accumulates history. After a sudden context change, information from the old chapter or market conditions continues influencing learning of the new context, causing interference.

Required mechanism: a "reset" that selectively clears obsolete, irrelevant memories when a logical discontinuity is detected, better initializing learning in the new context.

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
