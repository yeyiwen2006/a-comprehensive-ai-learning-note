---
title: "20.5 Self-Distillation"
chapter_title: "Optimizing Large-Model Architectures and Training Methods"
section_id: "20-05"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/20.大模型的架构和训练方法优化/20.5 自蒸馏.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 20.5 Self-Distillation

## I. An Expert-Demonstration-Based Method: Self-Distillation Fine-Tuning (SDFT)

1. Background

Currently, supervised fine-tuning (SFT) is the mainstream paradigm for learning from expert demonstrations. SFT is off-policy learning. When a model must adapt sequentially to new tasks or domains, it causes severe degradation in generalization and catastrophic forgetting. Although on-policy RL can significantly reduce forgetting and improve out-of-distribution generalization, it usually requires an explicit reward function. In many real-world scenarios, constructing such a precise reward function is very difficult or even infeasible.

2. Core principles

To obtain the advantages of on-policy learning when only demonstration or environmental-feedback data are available, without an explicit reward function, self-distillation lets the same model play both "teacher" and "student" during training.

The student receives only the original task prompt and generates autoregressively. The teacher receives the task prompt and expert demonstration, using the large model's in-context learning ability to generate the $i$th token embodying the correct task intent from the student's first $i-1$ tokens, providing a high-quality guidance signal.

The distributions corresponding to the student and teacher can be written as:

- Student: given prefix $y_{<t}$ and original question $x$, predict the probability distribution of the next token, $\pi_\theta(\cdot \mid y_{<t}, x)$.
- Teacher: given the same prefix and an additional expert-demonstration-enhanced prompt $c$, provide a higher-quality probability distribution, $\pi(\cdot \mid y_{<t}, x, c)$.

The overall sequence-level reverse KL optimization objective is:

$$
\mathcal{L}(\theta)
= D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid x)\,\Vert\,\pi(\cdot \mid x,c)\right)
= \mathbb{E}_{y\sim \pi_\theta(y\mid x)}
\left[
\log\frac{\pi_\theta(y\mid x)}{\pi(y\mid x,c)}
\right].
$$

The outer expectation $\mathbb{E}_{y\sim \pi_\theta(y\mid x)}$ indicates that the sampled data distribution comes from the student policy $\pi_\theta$.

This actually means summing over all possible next tokens v in the entire vocabulary V:

$$
\nabla_\theta D_{\mathrm{KL}}(\pi_\theta\Vert\pi)
=
\sum_{v\in V}
\pi_\theta(v\mid y_{<t})
\nabla_\theta \log \pi_\theta(v\mid y_{<t})
\left(
\log\frac{\pi_\theta(v\mid y_{<t})}{\pi(v\mid y_{<t},c)}
+ 1
\right).
$$

When writing the vocabulary sum at each time step as an analytical gradient estimate, the probability weights of the student distribution must be retained:

$$
\hat{g}_{\mathrm{analytic}}
=
\sum_{t=1}^{T}\sum_{v\in V}
\pi_\theta(v\mid y_{<t},x)
\nabla_\theta \log \pi_\theta(v\mid y_{<t},x)
\left(
\log\frac{\pi_\theta(v\mid y_{<t},x)}{\pi(v\mid y_{<t},x,c)}
+1
\right).
$$

In practical terms, this means that whenever the student reaches a step, it examines how much its probability distribution differs from the teacher's across every possible option in the entire vocabulary at that step, then reduces this difference.

In addition, the "expert demonstration" need not come from an external source: it can also be a correct answer obtained through the model's own later revisions. Analogous to humans consolidating memories during sleep, the model can, asynchronously with the main process, use correct answers and successful experiences it later obtains for the same problem to correct its initial erroneous output produced with less context.

## II. An Environmental-Feedback-Based Method: Self-Distillation Policy Optimization (SDPO)

1. Background

In RL training, we usually provide only a sparse scalar outcome (such as "pass: 1" or "fail: 0"), obscuring the rich underlying state information. The algorithm can only apply indiscriminate gradient penalties to all tokens in an entire piece of code, making it difficult for the model to identify exactly where it went wrong: a syntax error on a particular line, a flaw in logical reasoning, or merely a mishandled boundary condition. We want denser feedback.

2. Core principles

The student autoregressively generates an attempted solution based only on the original instruction prompt. The attempt is executed in a verifiable environment, which returns not only a score but also detailed, rich feedback (such as error messages). Based on the original instruction and environmental feedback, a teacher with identical weights uses strong in-context learning capabilities to predict the correct next-token distribution at each position of the student's generated sequence. It reflects on the environmental feedback it sees, thereby correcting the student's erroneous output. The student approximates this distribution by minimizing KL divergence (the same reverse KL introduced earlier).

3. Benefits of SDPO

Compared with RL using sparse reward signals, SDPO can precisely calculate a large gradient for the specific token causing an error, guiding the model to correct only the genuinely erroneous part. The sparse, single scalar reward originally available only at the end of the sequence is effectively transformed into a dense guidance signal at every token-generation step. Theoretical analysis proves that this mechanism is equivalent to optimizing an implicit teacher-defined reward within the maximum-entropy RL framework.

This design, which can calculate dense gradients for an entire sequence with only one .forward() call in PyTorch, not only avoids off-policy distribution shift but also far surpasses traditional methods in which two models separately generate sequences of different lengths in GPU-memory utilization and computational efficiency.

## References

- Shenfeld, I., Damani, M., Hubotter, J., & Agrawal, P. (2026). [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897). arXiv:2601.19897.
- Hübotter, J., Lübeck, F., Behric, L., et al. (2026). [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802). arXiv:2601.20802.
