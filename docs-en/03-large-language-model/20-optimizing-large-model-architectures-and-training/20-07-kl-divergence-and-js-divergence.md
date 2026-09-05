---
title: "20.7 KL Divergence and JS Divergence"
chapter_title: "Optimizing Large-Model Architectures and Training Methods"
section_id: "20-07"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/20.大模型的架构和训练方法优化/20.7 KL散度与JSD散度.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 20.7 KL Divergence and JS Divergence

In large language models, KL divergence measures differences between output probability distributions. It is calculated over a vocabulary of size $V$, not over the hidden-layer dimensions before the output, because the model's outputs are fundamentally discrete tokens.

$$
D(P\Vert Q)=\sum_{i=1}^{V}p_i\log\frac{p_i}{q_i}
$$

Note: calculating perplexity averages the negative log-probabilities of all tokens in a sequence, whereas calculating KL divergence averages the KL divergence over all words in the vocabulary when outputting one token. The two use different forms of averaging.

In research, JS divergence, a variant of KL divergence, is often used for greater numerical stability. It takes the average $M$ of $P$ and $Q$:

$$
M=\frac{1}{2}(P+Q)
$$

JS divergence is defined as:

$$
\mathrm{JSD}(P,Q)=\frac{1}{2}\left(D_{\mathrm{KL}}(P\Vert M)+D_{\mathrm{KL}}(Q\Vert M)\right)
$$

Its function is similar to KL divergence, but the denominator inside $\log$ does not become zero. For example, studying the JS divergence between neuron activation values in a model's intermediate hidden layers and output layer (projecting first if their dimensions differ) can determine the degree of similarity between their distributions.

For an LLM, the KL divergence of an autoregressive output of $n$ tokens is:

$$
D_{\mathrm{KL}}(P(Y)\Vert Q(Y))
=
\sum_Y P(Y)\log\frac{P(Y)}{Q(Y)}.
$$

Because directly summing over all possible sequences is infeasible (there are $V^n$ possibilities), the autoregressive property must be used to decompose it:

$$
D_{\mathrm{KL}}(P(Y)\Vert Q(Y))
=
\mathbb{E}_{Y\sim P}
\left[
\sum_{t=1}^{n}
\left(
\log P(y_t\mid y_{<t})-\log Q(y_t\mid y_{<t})
\right)
\right]
=
\sum_{t=1}^{n}
\mathbb{E}_{Y\sim P}
\left[
\log\frac{P(y_t\mid y_{<t})}{Q(y_t\mid y_{<t})}
\right].
$$

The difficulty here lies in $Y\sim P$, which requires considering the probability distribution of the entire sequence. Notice that the conditional probabilities on the right are conditioned on $y_{<t}$. Given a known distribution of $y_{<t}$, the autoregressive property also determines the distribution of $y_t$, so the distribution of $Y=\{y_1,\ldots,y_t\}$ is known as well. We can therefore rewrite the expression as:

$$
D_{\mathrm{KL}}(P(Y)\Vert Q(Y))
=
\sum_{t=1}^{n}
\mathbb{E}_{y_{<t}\sim P(y_{<t})}
\left[
D_{\mathrm{KL}}\left(P(y_t\mid y_{<t})\Vert Q(y_t\mid y_{<t})\right)
\right].
$$

This means that the KL divergence between two sequence distributions equals the sum of the expected KL divergences between their conditional probability distributions at each step (where the expectation is calculated over histories generated from distribution $P$).

## References

- Kullback, S., & Leibler, R. A. (1951). [On Information and Sufficiency](https://doi.org/10.1214/aoms/1177729694). The Annals of Mathematical Statistics, 22(1), 79-86.
- Lin, J. (1991). [Divergence measures based on the Shannon entropy](https://doi.org/10.1109/18.61115). IEEE Transactions on Information Theory, 37(1), 145-151.
