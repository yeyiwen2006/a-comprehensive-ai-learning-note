---
title: "20.8 Perplexity"
chapter_title: "Optimizing Large-Model Architectures and Training Methods"
section_id: "20-08"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/20.大模型的架构和训练方法优化/20.8 困惑度.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 20.8 Perplexity

Perplexity (PPL) is the gold-standard metric for evaluating language models. Its core idea is that a good model should assign a high predicted probability to the true next word. The more "perplexed" the model is, the less accurate its prediction and the higher its perplexity.

## I. Intuitive Understanding

Suppose a perfect language model is always 100% certain of the next word (its predicted probability is $1$). Its perplexity is then $\exp(0)=1$. This means the model has no uncertainty, equivalent to having only 1 word to choose from each time (the correct one).

For a very poor model, such as one that always considers all 100,000 vocabulary words equally probable (each with probability $1/100000$), the perplexity is 100,000. This means it is highly perplexed, equivalent to choosing among 100,000 words every time.

A good model's perplexity should lie between 1 and the vocabulary size. For example, modern models can reach a perplexity of around 20 on WikiText-2. This means that, on average, the model chooses one word from 20 each time and is relatively "certain" about what comes next.

## II. Mathematical Formulation and Derivation

Perplexity is closely related to cross-entropy in information theory. Cross-entropy measures the difference between the model's predicted probability distribution and the true distribution (here, actually a one-hot distribution, with probability 1 for the true word and 0 for all others). In fact, exponentiating the cross-entropy loss gives the perplexity of the entire sequence ("the average number of tokens to choose from at each step when generating this sequence"). The reciprocal of perplexity raised to the nth power is the joint probability of generating the sequence. Minimizing perplexity is mathematically equivalent to maximizing the joint probability the model assigns to the entire sequence.

Mathematical derivation:

1. Starting point: sequence probability

A language model aims to assign a high probability to the true sequence $W=(w_1,w_2,\ldots,w_N)$. By the chain rule, this probability is the product of all conditional probabilities:

$$
P(W)=P(w_1)\prod_{t=2}^{N}P(w_t\mid w_1,\ldots,w_{t-1})
$$

2. Take the logarithm (to turn products into sums)

Directly optimizing the product is mathematically troublesome (it may underflow). We take its logarithm, converting it into a log-likelihood. Because the logarithm is monotonic, maximizing the product of probabilities is equivalent to maximizing the sum of log-likelihoods.

$$
\log P(W)=\log P(w_1)+\sum_{t=2}^{N}\log P(w_t\mid w_1,\ldots,w_{t-1})
$$

3. Take the average (to remove the effect of length)

To compare performance across sequences of different lengths, take the average to obtain the average log-likelihood.

$$
\frac{1}{N}\log P(W)
$$

4. Take the negative (to obtain a loss)

We normally minimize a "loss" function rather than maximize a "score" function. Taking the negative gives the average negative log-likelihood. Lower values are better.

$$
L=-\frac{1}{N}\log P(W)
$$

5. Definition of perplexity

Perplexity is defined as the exponential of this average negative log-likelihood:

$$
\mathrm{PPL}=\exp(L)=\exp\left(-\frac{1}{N}\log P(W)\right)
$$

By the laws of exponents, this can be rewritten as:

$$
\mathrm{PPL}=P(W)^{-1/N}
=\left[P(w_1)\prod_{t=2}^{N}P(w_t\mid w_1,\ldots,w_{t-1})\right]^{-1/N}
$$

6. Relationship to probability

Increasing $P(W)^{1/N}$ increases the sequence's geometric mean probability. Therefore, minimizing perplexity is mathematically equivalent to maximizing the joint probability $P(W)$ of the entire sequence.

## III. Information-Theoretic Explanation

Step 1: understand the meaning of "taking the logarithm"—measuring surprisal

In information theory, $-\log_2 P(\mathrm{event})$ measures an event's "surprisal" or "information content."

If an event is certain to occur (probability $P=1$), its surprisal is $-\log_2 1=0$ bits. You are not surprised at all.

If an event is extremely rare (probability $P\to 0$), its surprisal approaches infinity. You are extremely surprised.

If an event has a 50% probability of occurring ($P=0.5$), its surprisal is $-\log_2 0.5=1$ bit.

In language models, $-\log P(w_t\mid \mathrm{context})$ measures how "surprised" the model is when the true next word $w_t$ appears. The lower the probability, the higher the surprisal.

Thus, the average negative log-likelihood L is:

$$
L=\frac{1}{N}\sum_{t=1}^{N}-\log P(w_t\mid w_{<t})
$$

Its practical meaning is the model's average "degree of surprise" when predicting each word in the sequence (in bits per word).

Step 2: understand the meaning of "exponentiation"—returning to the choice space

We now have an "average surprisal" (for example, 6.5 bits per word). But this number is not intuitive enough. How surprising is 6.5 bits of surprise?

If $L$ uses natural logarithms, the core role of $\exp(L)$ is to convert "surprisal" back to "probability space"; if $L$ uses base-2 logarithms, the corresponding conversion is $2^L$. More specifically, it converts average surprisal into an equivalent choice problem under a uniform distribution.

Consider an example:

Suppose a very poor model predicts the next word completely at random, with a vocabulary $|V|$ of 1024 words. Its predicted probability for each word is $1/1024$.

The surprisal of each word is:

$$
-\log_2(1/1024)=\log_2(1024)=10
$$

The average surprisal $L$ is also 10 bits per word.

Calculate perplexity:

$$
\mathrm{PPL}=2^L=2^{10}=1024
$$

An important point is that information theory often uses 2 as the base of $\log$, while mathematical calculations often use natural logarithms (base $e$). The two differ only by a constant factor; consistency is what matters. This worst model's perplexity is exactly the vocabulary size, 1024.

This can be interpreted as follows: when predicting each word, the model behaves as if it were blindly guessing from a uniformly distributed vocabulary of 1024 candidate words. Its uncertainty is equivalent to choosing 1 out of 1024.

As another example, suppose a good model has an average negative log-likelihood of $L=\log 64$ (approximately 4.16), calculated using natural logarithms.

Calculate perplexity:

$$
\mathrm{PPL}=\exp(L)=\exp(\log 64)=64
$$

This can be interpreted as follows: although the actual vocabulary may contain thousands or tens of thousands of words, the model is sufficiently capable that its average uncertainty when predicting each word is equivalent to choosing from a small set of only 64 candidates.

## References

- Jurafsky, D., & Martin, J. H. (2026). [Speech and Language Processing, Chapter 3: N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf). Draft of August 19, 2026.
- Brown, P. F., Della Pietra, V. J., deSouza, P. V., Lai, J. C., & Mercer, R. L. (1992). [Class-Based n-gram Models of Natural Language](https://aclanthology.org/J92-4003/). Computational Linguistics, 18(4), 467-480.
