---
title: "17.3 Expert Iteration and Rejection Sampling"
chapter_title: "Reinforcement Fine-Tuning"
section_id: "17-03"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/17.强化微调/17.3 专家迭代与拒绝采样.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 17.3 Expert Iteration and Rejection Sampling

## I. Basic Idea of Expert Iteration

Expert iteration involves two roles that improve each other in a loop:

1. Apprentice: the neural network being trained (policy $\pi_\theta$). It performs fast inference, but decision quality is limited by its current weights.
2. Expert: usually not an external human or stronger model, but a search operator built on the current apprentice's capabilities. Examples include the current policy combined with Monte Carlo tree search (MCTS), or with self-sampling, reranking, and verifiers. The expert is computationally expensive and slow, but multi-step search or verification produces higher-quality decisions than the apprentice alone.

The essence is that the apprentice uses imitation learning (SFT) to learn to make decisions like the expert, obtaining high-quality results quickly without search. In the next iteration, the stronger apprentice supports deeper expert search.

Stage I is expert improvement through data generation. With parameters $\theta$ fixed, a more computationally intensive expert algorithm generates data better than the current model's direct output.

Let the current state be prompt $x$ and the model policy be $\pi_\theta$. Define an expert operator $\Phi$ that improves on $\pi_\theta$:

$$
\pi_{expert}(y|x)=\Phi(\pi_\theta,x)
$$

For LLM reasoning, $\Phi$ is commonly implemented through:

- Search: sample $N$ times for the same $x$ (majority voting), or use beam search.
- Verification: use unit tests for code or a reinforcement-learning scorer to identify correct output paths.
- Dataset construction (MCTS): let AlphaZero construct a search tree to evaluate action values $Q(s,a)$ and identify the highest.

The result is high-quality trajectory data $\mathcal{D}_{new}=\{(x,y_{expert})\}$, where $y_{expert}$ is a correct reasoning path obtained through search and verification.

Stage II is imitation learning/distillation. Treat $\mathcal{D}_{new}$ from stage I as ground truth and update model parameters through supervised learning (SFT).

The objective minimizes KL divergence, that is, cross-entropy, making the model's “intuition” $\pi_\theta$ approximate the post-search expert distribution $\pi_{expert}$:

$$
\theta_{new}=\arg\min_\theta \mathbb{E}_{x\sim D}\left[-\log\pi_\theta(y_{expert}|x)\right]
$$

AlphaZero fits both actions (the policy head) and winning probabilities (the value head). For LLM reasoning, the mainstream approach currently fits only the generated token sequence (the policy).

Although the parameter update looks like supervised learning, it performs policy improvement and is actually reinforcement learning under a given environment and reward. The earlier MCTS discussion also introduced expert-iteration updates that do not use supervised fine-tuning.

A key mathematical principle is that multi-step search generally outperforms single-step prediction.

Let $V(x)$ be the true value of state $x$ (whether the problem can be solved). Apprentice policy $\pi_\theta$ is a parameterized approximation to the optimal policy; expert policy $\pi_{expert}$ improves $\pi_\theta$ through lookahead.

From properties of the Bellman equation, if $\Phi$ is an improvement operator with respect to the value function:

$$
\mathrm{Performance}(\Phi(\pi_\theta))\ge \mathrm{Performance}(\pi_\theta)
$$

ExIt training proceeds as:

1. $\pi_{t+1}\approx\Phi(\pi_t)$ (policy update: the apprentice catches up with the expert).
2. $\pi_{expert}=\Phi(\pi_{t+1})$ (the expert improves again using the new apprentice).

Theoretically, this upward cycle can eventually converge to the task's optimal solution.

## II. Rejection Sampling Fine-Tuning (RFT)

Once an SFT model has some capability, using its own generated data to improve it further is called expert iteration.

Principle: use the SFT model to generate $k$ different reasoning paths $\{y_1,y_2,\ldots,y_k\}$ for one question $x$. An external rule-based checker, such as a Python interpreter or answer matcher, filters for correct results. These model-generated correct paths are added to the training set for further fine-tuning.

Workflow:

1. Rollout: for question $x$ in the dataset, sample $k$ solutions from current policy $\pi_\theta$, with higher temperature for diversity.
2. Evaluation: for mathematics, extract the final answer and compare it with ground truth through exact matching; for code, run unit tests.
3. Filtering and construction: retain verified samples $(x,y_{correct})$.
4. Retraining: mix the new samples into the original SFT data and run another round of SFT.
5. Iteration: repeat.

## III. Self-Taught Reasoner (STaR): Learning from Reasoning Corrected After Errors

The model generates a reasoning process and answer. If the answer is wrong, inject the correct answer as a hint and ask for a rationalization that derives the correct answer. Add these retrospectively corrected reasoning examples to the training set for supervised fine-tuning.

## IV. Applications of Expert Iteration

1. LLMs

If PPO will later optimize reasoning, several expert-iteration rounds usually come first. PPO is sensitive to the initial policy, so the model's distribution must first reach a sufficiently high level (an SFT model is often still too weak), after which PPO can optimize extremely difficult cases or perform alignment. Expert iteration suits this role. GRPO, in contrast, updates through within-group relative advantages: a few correct samples produce large positive gradients. Each round reinforces the model's own good samples and suppresses bad ones, effectively performing expert iteration and avoiding the extra computation of evaluating search results.

2. AlphaZero

Original MCTS, rather than the LLM-specific variant, explores possible paths. Cross-entropy brings policy-network outputs closer to the MCTS sampling results. The value network's target is directly the final win/loss outcome (only a final reward of 1 or -1), so its prediction v should accurately predict final result z.

Given historical state $s_t$, the network's predicted $v_t$ should approach final outcome $z_t$.

AlphaZero's total loss has value, policy, and regularization terms. Value updates use mean squared error (MSE):

$$
l_{\mathrm{value}} = (z - v)^2
$$

The complete loss is:

$$
\mathcal{L} = \underbrace{(z - v)^2}_{\text{value loss}}
- \underbrace{\pi^{T}\log p}_{\text{policy loss}}
+ \underbrace{c\|\theta\|^2}_{\text{regularization}}
$$

- $z$: the true final winner of the game.
- $v$: the network's predicted chance of winning at state $s$.

One might ask: if MCTS's $Q$ is more accurate than the network's direct prediction $v$, why not use $Q$ as the target for updating $v$?

This is a reasonable intuition, but in AlphaZero:

- The policy network fits the MCTS search probabilities $\pi$, reflecting which move MCTS considers better.
- The value network fits the true outcome $z$.

The reason is that, although MCTS's $Q$ is more accurate than $v$, it still aggregates network predictions of $v$ and is biased. Fitting $Q$ would amount to a circular dependence and could accumulate bias. The true outcome $z$ is unbiased, although its variance is high (winning a game does not mean a particular move was good). With enough data, the network can learn the most accurate position evaluation.

## References

- Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404). Science.
- Anthony, T., Tian, Z., & Barber, D. (2017). [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439). arXiv:1705.08439.
- Zelikman, E., Wu, Y., Mu, J., & Goodman, N. (2022). [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465). NeurIPS 2022.
- Silver, D., Huang, A., Maddison, C. J., et al. (2016). [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961). Nature.
