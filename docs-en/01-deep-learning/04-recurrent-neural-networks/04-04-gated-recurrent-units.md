---
title: "4.4 Gated Recurrent Units (GRUs)"
chapter_title: "Recurrent Neural Networks"
section_id: "04-04"
language: en
source_language: zh
source_docx: "第1部分 深度学习/4.循环神经网络/4.4 门控记忆单元（GRU）.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 4.4 Gated Recurrent Units (GRUs)

## I. Basic Principles

1. Reset gate R_t: controls how much of the past state we "might still want to remember." It determines how much past information to "forget" (such as useless webpage HTML code that should not participate in the candidate-state computation) before combining it with the new input to form a candidate state.

The reset gate helps capture short-term dependencies. By controlling how much history participates in the candidate-state computation, it influences the immediate response at the current time. This is effective for understanding relationships between adjacent tokens (short-term patterns).

2. Update gate Z_t: controls the proportions of the old and candidate states in the new state. It decides how much past information is retained directly (such as information that must remain unchanged).

The update gate helps capture long-term dependencies. By controlling the information path, it allows the network to copy past states directly. If the network learns to keep the update gate close to 1 for a long period, distant information can pass almost losslessly into the future, perfectly addressing vanishing gradients and capturing long-term patterns.

Rather than computing the new hidden state directly, the GRU first computes a candidate hidden state $\tilde{H}_t$. It represents the state that could form from the current input and filtered past information.

Reset gate $R_t$ plays a key role: elementwise multiplication (the Hadamard product) with the preceding state $H_{t-1}$ "resets" or "filters" history.

- If an element of $R_t$ is close to 1, the corresponding past information passes through, reducing the formula to an update resembling a conventional RNN: $\tanh(X_t W_{xh} + H_{t-1}W_{hh} + b_h)$.
- If an element of $R_t$ is close to 0, the corresponding past information is "masked" or "forgotten." The candidate then depends only on current input $X_t$ (like a simple MLP), helping ignore irrelevant information.

The formula is:

$$
\tilde{H}_t = \tanh(X_t W_{xh} + (R_t \odot H_{t-1})W_{hh} + b_h)
$$

Here, $\odot$ is the Hadamard product (elementwise multiplication), and $\tanh$ keeps candidate values within $(-1, 1)$.

Finally, new hidden state $H_t$ is a convex combination of old state $H_{t-1}$ and candidate $\tilde{H}_t$, controlled by update gate $Z_t$.

- If an element of $Z_t$ is close to 1, the new state largely retains the old information and almost ignores the candidate (so the current input has little effect). The network can skip unimportant time steps and effectively pass on distant information, addressing long-term dependencies.
- If an element of $Z_t$ is close to 0, the new state favors the candidate and focuses more on the current input.

The GRU can thus flexibly choose between "remembering" long-term information and "attending to" short-term input.

The formula is:

$$
H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H}_t
$$

An intuitive interpretation:

A GRU resembles an efficient modern workstation:

Reset gate: clear irrelevant material from the whiteboard before considering a new proposal (~h_t).

Update gate: decide how much old content to erase and how much of the new proposal to write on the final whiteboard (h_t).

Two questions to consider:

1. Why use separate gates instead of directly adjusting weights?

Gates have an important property: "elementwise multiplication."

In ordinary matrix multiplication, adjusting a weight necessarily affects all elements multiplied by it, making it difficult to retain or erase information at a particular position. Gate elements, by contrast, correspond one-to-one with neurons.

2. Why can the two gates not replace each other?

With only a reset gate and no update gate, the candidate-state function struggles to fit the identity y=x, so unchanged information transmission is not guaranteed. With only an update gate and no reset gate, irrelevant information at individual positions influences the candidate (and, for the reasons discussed earlier, the corresponding weights cannot simply be set to 0).

## II. Computing the Reset and Update Gates

At a given time step $t$:

$$
R_t = \sigma(X_t W_{xr} + H_{t-1}W_{hr} + b_r)
$$

$$
Z_t = \sigma(X_t W_{xz} + H_{t-1}W_{hz} + b_z)
$$

where:

- $\sigma$ is the sigmoid activation.
- $W_{xr}, W_{xz}$ are weight matrices multiplying current input $X_t$.
- $W_{hr}, W_{hz}$ are weight matrices multiplying previous hidden state $H_{t-1}$.
- $b_r, b_z$ are biases.

Why compute both gates from current input X_t and previous hidden state H_{t-1} through a fully connected layer followed by sigmoid?

1. Purpose: gates must be "context-aware" decision-makers

The gates must make two key decisions at every time:

Reset gate: to process this new input properly, how much background information that I was remembering should I forget?

Update gate: should I mostly remember the old information or mostly trust the new?

These cannot be rigid rules. They must adapt to the current word and context.

For example, in sentiment analysis:
Consider "This movie is not good at all, it is actually terribly boring."

When the model reads "not":

The reset gate should recognize that the next word, "good," will have its meaning reversed. It must weaken/reset the positive expectation created by "is" and prepare for negated information.

The update gate should recognize "not" as an extremely important signal, retain that state firmly, and pass it onward strongly to influence judgments of subsequent words.

If gates were fixed or depended only on the current word, they could not make these fine-grained, crucial decisions based on the entire sentence's context (encoded in the previous hidden state).

Gate decisions must therefore use two information sources:

Current observation (X_t): what word am I seeing now?

Historical context (H_{t-1}): what have I read, and what is the current background of my understanding?

Only their combination enables the most informed, context-aware decisions.

2. Structure: letting the model learn the trade-off

Multiplying the current input and past state by separate weights and adding them (an "affine transformation") lets the model learn from training data when to trust the current input more and when to depend more on historical memory.

For words that can overturn meaning (such as "not" and "but"), it learns to give the current input large weight so that the current word dominates gate opening and closing.

For unimportant words (such as "the" and "a"), it learns to let the historical state dominate, instructing the gates: "Ignore it and preserve the status quo!"

This design gives the model substantial flexibility.

3. Activation: Sigmoid, a perfect "proportion regulator"

Sigmoid is a remarkable "compressor," transforming any number into a value between 0 and 1.

Natural interpretability: values between 0 and 1 represent "percentage opening." For example, 0 is "fully closed," 1 is "fully open," and 0.7 is "70% open."

Trainability: the function is smooth, which is essential for training through backpropagation. Errors can pass through smoothly to adjust weights.

Clear decisions: although sigmoid saturates in extreme cases (causing vanishing gradients), this is advantageous for gating. Gate outputs can approach 0 or 1 arbitrarily closely, enabling decisive "open" or "closed" behavior and stable memory flow.

## III. Summary

By coordinating reset and update gates, a GRU dynamically and selectively manages memory, achieving strong results across sequence-modeling tasks.

## References

- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). [Dive into Deep Learning](https://D2L.ai). Cambridge University Press.
- Cho, K., van Merrienboer, B., Gulcehre, C., et al. (2014). [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078). EMNLP 2014.
