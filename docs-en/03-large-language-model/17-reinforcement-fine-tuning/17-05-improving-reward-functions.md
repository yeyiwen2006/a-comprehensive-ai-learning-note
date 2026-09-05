---
title: "17.5 Improving Reward Functions"
chapter_title: "Reinforcement Fine-Tuning"
section_id: "17-05"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/17.强化微调/17.5 奖励函数的改进.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 17.5 Improving Reward Functions

## I. Addressing Sparse Rewards

RLVR relies on verifiable reward signals, such as mathematical answers or programming outputs, which are very sparse. An agent receives a nonzero score only when its generated code runs perfectly from beginning to end: loading data, training, predicting, and producing a submission file. A program failing at data loading and one almost succeeding but failing only when saving the final file both receive zero reward (or the same fixed negative failure value). This makes useful gradients difficult to obtain, especially on hard tasks. The agent cannot readily distinguish “completely wrong” from “nearly correct,” reducing learning efficiency. Possible improvements include:

1. Combining step decomposition with dense feedback

The core idea is to give each AI operation a “progress indicator” and have it output intermediate variables, allowing denser rewards so that early training is not entirely without gradients.

Note, however, that once accuracy reaches a certain level, training generally switches back to pure outcome rewards so the AI explores fully correct answers rather than stopping at local optima.

(1) Intermediate-step outputs (hard scoring)

Automatic code injection: before running agent-generated code, use an independent, fixed language model to insert intermediate print statements such as print("loaded data") and print("trained model").

Parse outputs for scoring: after execution, scan terminal logs with regular expressions to detect successfully printed intermediate variables.

Graded rewards: each correctly output intermediate variable earns a small reward, such as +0.1.

Advantage: even if the program ultimately fails, the agent receives informative feedback, gradually learning data loading, then model construction, and ultimately task completion. These rewards are also generally objective and verifiable.

(2) Scoring the reasoning process (LLM scoring)

Do not rely exclusively on outcomes; consider the reasoning. A sound chain of thought with a calculation error can still receive partial credit. This resembles grading extended mathematics problems in China's college entrance examination: a wrong answer can still earn most points if the reasoning is correct, while a bare answer without reasoning cannot.

Process scoring can include structural checks: mathematics tasks can be checked for key intermediate variables and reconstruction steps, while programming tasks can be checked for required function and variable names.

This should nevertheless be used cautiously, because it is not RLVR with an unambiguous standard. LLM generalization is not unlimited. When evaluation is subjective or flexible, such as scoring reasoning rather than answers alone, an RL agent is likely to discover reward hacking, and not every vulnerability to adversarial examples can be perfectly eliminated.

2. Why process rewards require caution, such as use only early in training

Although the preceding methods ease training, reward sparsity is intrinsic to many real-world problems and environments (biological evolution, for example, rewards surviving population counts). Artificially making rewards denser cannot fundamentally avoid changing the optimization objective. An excellent method humans do not understand, such as AlphaGo's Move 37, might even be incorrectly penalized.

Overly dense rewards can also encourage excessive attention to immediate reward and local optima, weakening performance on tasks requiring long-term planning and consistency. Similarly, evaluating researchers too frequently can favor quick, short-term studies and reduce the chance of major discoveries. Some reward sparsity is inherently more fundamental.

Thus, process rewards are generally used only early to address sparsity. Once reward reaches a threshold, training switches to pure outcome rewards.

2. Supervised reinforcement learning (SRL) (Google, 2025)

For a less capable model, forcing rote memorization of high-quality answers through SFT can make it less capable. Instead, like a patient tutor, a system can break a difficult problem into small steps and assign a similarity score against the reference answer after every step. This combination of guidance and encouragement can teach small models complex reasoning they otherwise could not master.

SRL decomposes a complete solution into key actions instead of treating it as one unit. Before each action, the model generates an “inner monologue” inside `<think>` tags. This separates flexible reasoning from the structured execution steps being evaluated.

After each action, the system compares it textually with the corresponding expert-demonstration action, producing a reward between 0 and 1. An imperfect step can still earn positive feedback if broadly headed in the right direction. This “close enough” similarity reward provides a smooth, continuous learning signal through a complex exploration space.

To improve efficiency, SRL filters training examples for which almost all attempts have similar similarity scores and therefore weak learning signals.

This method is generally used early in reinforcement fine-tuning or when training small models with large models. Later RL stages for large models still use RLVR to elicit more potential and discover new methods.

3. Goal-oriented reinforcement learning

The goal-oriented reinforcement learning section provides details. Here are examples in reinforcement fine-tuning.

Mathematical problem solving:

- Original goal $g$: calculate $\int x^2 dx$.
- Model behavior: incorrectly obtains $\frac{1}{2}x^2$, perhaps forgetting coefficient $\frac{1}{3}$ or differentiating with respect to $x$.
- Traditional approach: mark it wrong, reward 0.
- Goal-oriented approach:
  - Analyze the output $\frac{1}{2}x^2$.
  - Reverse reasoning or relabeling: it is the correct answer to “calculate $\int x dx$,” or “differentiate $\frac{1}{6}x^3$.”
  - Update: tell the model that its derivation was perfect for solving $\int x dx$.
  - Effect: it learns correct integration logic, which it had initially applied in the wrong place.

Code generation:

- Original goal $g$: load data and train a model.
- Model behavior: loads data successfully but crashes during training.
- Relabeled goal $g'$: “write a data-loading script.”
- Result: the new trajectory receives full reward, strengthening correct data loading.

4. Starting reasoning from intermediate states

During RLVR training, models often produce groups of reasoning attempts close to success. HiPO extracts intermediate states from successful trajectories within these near-miss groups as hints, letting some new trajectories continue conditioned on the original problem and hint. This aims to increase within-group outcome diversity and alleviate zero relative advantages when every sample fails. It can increase the opportunity for nonzero learning signals, but does not guarantee that each hint improves the final policy.

5. Improving RL algorithms

Better RL algorithms can propagate sparse rewards more effectively. GRPO, for example, outperforms PPO in this respect, since PPO's critic may train very unstably under sparse rewards. Such improvements help models adapt to the intrinsically sparse rewards common in the real world.

## II. Adjusting Answer Format

1. HERO: combining outcome and format rewards (Meta FAIR, 2025)

First determine correctness: a binary 0–1 verifier splits answers into correct and incorrect groups. Hierarchical normalization groups all outputs by correctness and keeps reward scores within their boundaries. Then quantify quality: a reward model finely ranks answers within each group, assigning more granular quality scores. Variance-aware weighting focuses training on the most informative difficult problems. A poorly formatted correct answer therefore scores above a well-formatted incorrect answer, keeping training pointed in the correct direction.

## III. Analyzing Output Equivalence

For mathematical derivations, especially expression simplification or problems with nonunique solution processes, string matching alone is insufficient. Rewards should reflect mathematical substance. Symbolic equivalence uses symbolic-computation tools such as Python's SymPy to determine whether two expressions are mathematically equivalent. Compared with exact matching against a reference answer, it generalizes better and more accurately reflects mathematical reasoning.

## IV. Preventing Cheating and Answer Guessing

Add appropriate process checks. In code-execution environments, prohibit importing certain dangerous functions such as eval and prohibit directly printing the answer.

## V. Redundant Outputs

Models often “overthink,” producing many redundant tokens for simple conclusions, repeatedly switching among incorrect paths, or constantly backtracking. This wastes valuable compute and increases inference latency.

Possible solutions:

1. Directly modifying the RL objective

Redundant tokens can be related to rewards, loss aggregation, and length normalization. An objective normalized by sequence length $T$ changes the gradient weights of samples of different lengths, so adjusting loss aggregation or length penalties may alleviate redundancy. However, the public DeepSeek-V3.2 report does not justify attributing redundant tokens solely to GRPO's $1/T$ normalization. Training data, sampling, decoding settings, and other factors also affect results, requiring specific ablations.

2. RePro (Shanghai AI Lab, 2025)

(1) Core observation: reasoning as optimization

RePro views a model's reasoning trajectory as a path searching a loss surface for an optimum.

Each reasoning step corresponds to a gradient update. The objective is to maximize the probability of generating the correct ground-truth answer. Good reasoning should substantially increase confidence in the correct answer at every step (decreasing loss), follow a firm direction, and avoid repeated switching or oscillation. Overthinking corresponds to saddle points, where many tokens barely improve answer probability, or gradient oscillation, where confused reasoning makes confidence fluctuate.

From this perspective, RePro designs process rewards embedded directly in RLVR procedures such as PPO and GRPO.

(2) Surrogate objective

RePro quantifies current confidence through the average log probability of the correct answer's tokens under the current reasoning context, similar to classification loss in supervised learning:

$$
\tilde{J}(\pi_\theta, q, \tau_{\le t}, a)
\triangleq
\frac{1}{|a|}\sum_{i=1}^{|a|}
\log \pi_\theta(a_{(i)} \mid q, \tau_{\le t})
$$

Before reasoning begins, directly guessing the answer has low probability and J̃ is small. As reasoning eliminates incorrect options and establishes the logical chain, confidence should increase and J̃ should rise. It should peak when reasoning reaches the conclusion. A higher value indicates greater confidence that the answer is correct, making it a reasonable optimization metric.

(3) Two scores derived from the J sequence

Compared with traditional RL's sparse outcome-only feedback, RePro introduces process-aware trajectory optimization. It separates changes in J̃ into two dimensions:

The magnitude score measures objective change: “How much closer did this reasoning segment bring the model to the answer?” In optimization, gradient magnitude determines descent speed. In reasoning, an effective chain of thought should markedly increase answer confidence. RePro measures this gain by comparing the post-step J̃ with baseline J̅, the confidence from answering directly without reasoning.

$$
\Delta(\pi_\theta, q, \tau_{\le t}, a)
=
\frac{
\tilde{J}(\pi_\theta, q, \tau_{\le t}, a) - \bar{J}_b(q)
}{
\bar{J}_b(q)
}
$$

$$
S_{\mathrm{magn},(t)}
\triangleq
\tanh\left(\Delta(\pi_\theta, q, \tau_{\le t}, a) + 1\right) + 1
$$

The tanh function normalizes scores to (0,1]. In practice, some steps can increase confidence exponentially, such as finally computing a crucial intermediate variable. Without a bound, the huge reward might cause exploding gradients or unstable training.

The stability score measures whether J rises smoothly. If J̃ is plotted as a curve, ideal reasoning rises monotonically. Fluctuations indicate self-doubt or logical confusion. RePro quantifies these fluctuations using Kendall's tau correlation:

$$
S_{\mathrm{stab},(t)}
=
\frac{
\sum_{i < j}
\mathrm{sign}(\tilde{J}_i - \tilde{J}_j)\cdot \mathrm{sign}(i - j)
}{
|\tau|(|\tau| - 1)
}
+
\frac{1}{2}
$$

This computes rank correlation between the J̃ sequence and time-step sequence {1,…,t}. High stability near 1 means each J̃ exceeds the previous value: progress at every step without retreat, corresponding to smooth movement along steepest descent. Low stability near 0 or negative values means a disordered sequence, taking two steps forward and one back or even showing severe logical regression. This corresponds to random movement around a saddle point, consuming steps (tokens) without substantive progress.

Weighted magnitude and stability scores form the final process score S, used to decide whether a reasoning path should be reinforced or penalized.

(4) Process-level rewards

RePro uses entropy-based selection:

Segmentation: split the reasoning chain into logical paragraphs {c1, c2,…, cN}, for example using blank lines \n\n.

Entropy calculation: calculate the entropy of each paragraph's first token, $\mathcal{H}(c_i^{(0)})$.

Top-k selection: compute RePro rewards only for the $k$ highest-entropy segments.

This substantially reduces cost, replacing whole-sequence calculation with $k$ points. It also concentrates guidance where the model is most uncertain and decisions most critical, remaining silent in confident, fluent low-entropy regions to avoid excessive intervention.

Then the process-score improvement $\Delta S$ serves as the segment's process-level reward. Combined with final correctness, it becomes input to RL advantage computation.

## References

- 温睦宁、林江浩、张伟楠、俞勇 (2025). [*Hands-on Learning of Large-Model Agents* (translated title; in Chinese)](https://haa.boyuai.com/). Posts & Telecom Press. ISBN 978-7-115-68638-1.
- Andrychowicz, M., Wolski, F., Ray, A., et al. (2017). [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495). NeurIPS 2017.
- Deng, Q., Chen, K., Zhang, M., & Xu, Z. (2026). [HiPO: Self-Hint Policy Optimization for RLVR](https://openreview.net/forum?id=rcb20pHmT1). ICLR 2026.
- DeepSeek-AI. (2025). [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556). arXiv:2512.02556.
- Deng, Y., Hsu, I.-H., Yan, J., et al. (2025). [Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](https://arxiv.org/abs/2510.25992). arXiv:2510.25992.
- Tao, L., Kulikov, I., Saha, S., et al. (2025). [Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dense](https://arxiv.org/abs/2510.07242). arXiv:2510.07242.
- Liu, J., Liu, H., Zhang, S., & Chen, K. (2025). [Rectifying LLM Thought from Lens of Optimization](https://arxiv.org/abs/2512.01925). arXiv:2512.01925.
