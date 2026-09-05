---
title: "11.1 Monte Carlo Methods"
chapter_title: "Value-Based Reinforcement Learning"
section_id: "11-01"
language: en
source_language: zh
source_docx: "第2部分 强化学习/11.基于价值的强化学习/11.1 蒙特卡洛方法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 11.1 Monte Carlo Methods

Value-based reinforcement learning learns a value function and chooses a policy from it. The key is how to solve for the value function.

Monte Carlo methods, also called statistical simulation methods, are numerical methods based on probability and statistics. They usually perform repeated random sampling and use statistical techniques to infer a numerical estimate of the desired quantity from the samples. A simple example is estimating a circle's area. Generate random points inside a square and count those falling inside the circle. The ratio of the circle's area to the square's area equals the ratio of points inside the circle to points inside the square. More random points produce an estimated area closer to the true circle area. Monte Carlo methods can estimate a policy's state-value function in a Markov decision process. A state's value is its expected return, so an intuitive approach is to sample many sequences under the policy in the MDP, calculate returns from that state, and take their expectation:

$$
V^\pi(s)=\mathbb{E}_\pi[G_t|S_t=s]\approx \frac{1}{N}\sum_{i=1}^{N}G_t^{(i)}
$$

A state may occur zero, one, or many times in a sequence. Monte Carlo value estimation calculates its return at every occurrence. Another option calculates only one return per sequence: use cumulative reward following the state's first occurrence, ignoring later occurrences. Suppose we sample sequences starting from state $s$ under policy $\pi$ to calculate state values. Maintain a counter and total return for each state, with the following procedure.

(1) Sample several sequences under policy $\pi$:

$$
s_0^{(i)} \xrightarrow{a_0^{(i)}} r_0^{(i)},s_1^{(i)} \xrightarrow{a_1^{(i)}} r_1^{(i)},s_2^{(i)} \xrightarrow{a_2^{(i)}} \cdots \xrightarrow{a_{T-1}^{(i)}} r_{T-1}^{(i)},s_T^{(i)}
$$

(2) For state $s$ at every time step of every sequence:

- Update the counter for state $s$: $N(s)\leftarrow N(s)+1$;
- Update the total return for state $s$: $M(s)\leftarrow M(s)+G_t$;

(3) Estimate each state's value as its average return, $V(s)=M(s)/N(s)$.

By the law of large numbers, as $N(s)\to\infty$, $V(s)\to V^\pi(s)$. Instead of summing all returns and dividing by their count, their expectation can also be calculated incrementally. For each state $s$ and corresponding return $G$:

- $N(s)\leftarrow N(s)+1$
- $V(s)\leftarrow V(s)+\frac{1}{N(s)}(G-V(s))$

## References

- [Hands-on Reinforcement Learning (translated title; in Chinese)](https://hrl.boyuai.com/).
