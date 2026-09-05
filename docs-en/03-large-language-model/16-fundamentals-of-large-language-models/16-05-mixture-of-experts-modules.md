---
title: "16.5 Mixture-of-Experts Modules"
chapter_title: "Fundamentals of Large Language Models"
section_id: "16-05"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/16.大语言模型的基本原理/16.5 混合专家模块.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 16.5 Mixture-of-Experts Modules

After the attention layer, tokens have extensively exchanged and integrated information. A feed-forward network (FFN) then processes it. The FFN is generally a gated GLU network:

A common SwiGLU FFN can be written as:

$$
FFN_{\mathrm{SwiGLU}}(x) = (\sigma(xW_1) \otimes (xV))W_2
$$

Here:

- $W_1$ is the gate branch's weight matrix, and $\sigma$ is an activation function such as Swish or SiLU.
- $V$ is the value branch's weight matrix.
- $\otimes$ denotes the element-wise Hadamard product.
- $W_2$ is the final output-projection weight matrix.

The algorithm proceeds as follows:

1. **Gate projection:** multiply input $x$ by $W_1$ to map it to the internal hidden dimension $d_{ff}'$, then apply activation $\sigma$ to produce a “gate” controlling information flow.
2. **Value projection:** in parallel, multiply $x$ by $V$ to map it to the same hidden dimension $d_{ff}'$. This branch stays linear, with no activation function.
3. **Gated fusion through element-wise multiplication:** multiply the nonlinear gate signal from step 1 element-wise by the linear value signal from step 2. This enables dynamic feature filtering, effectively allowing the input to decide which features to retain or amplify.
4. **Down-projection:** multiply the fused vector by $W_2$ to return to the original dimension $d_{model}$.

In practice, FFNs often account for a large proportion of parameters.

To expand parameter counts while controlling active computation per token, models such as DeepSeekMoE and Gemini 1.5 Pro use sparse MoE architectures. Typically, an MoE sublayer replaces the dense FFN in some or all Transformer blocks; exact layer counts and placements depend on each model's disclosed architecture. The router selects only a few experts based on token features rather than activating every expert's parameters, implementing sparse conditional computation.

## I. Assigning Inputs to Experts

### 1. Notation

Suppose the input contains $T$ tokens ($T=\text{Batch Size} \times \text{Sequence Length}$), with hidden dimension $d$.

- **Input tensor:** $X \in \mathbb{R}^{T \times d}$. Token $t$ is represented by row vector $x_t$.
- **Expert set:** $\mathcal{E}=\{E_1,E_2,\ldots,E_N\}$, containing $N$ expert networks.
- **Expert function:** $E_i(x): \mathbb{R}^d \to \mathbb{R}^d$, the $i$-th feed-forward network.
- **Gating parameters:** $W_g \in \mathbb{R}^{d \times N}$, the weight matrix used to calculate routing scores.

### 2. Gating and Sparse Routing

This is the MoE decision stage, selecting the $k$ experts to which each token is sent.

#### 2.1 Raw Logits

First calculate the raw matching-score matrix $H \in \mathbb{R}^{T \times N}$ between input $X$ and the experts:

$$
H = XW_g
$$

#### 2.2 Top-k Sparsification

For sparse computation, retain only the largest $k$ values in each row. Define $\mathrm{TopK}(v,k)$ to retain the $k$ largest entries of vector $v$ and set the rest to $-\infty$.

For token $t$'s score vector $h_t$ (row $t$ of $H$):

$$
\tilde{h}_t = \mathrm{TopK}(h_t,k)
$$

#### 2.3 Gating Weights

Normalize the sparsified scores with Softmax to obtain the final gating-weight matrix $G \in \mathbb{R}^{T \times N}$:

$$
G = \mathrm{Softmax}(\tilde{H})
$$

Note that $G$ is row-sparse: each row $G_t$ has only $k$ nonzero entries, representing the weights the token assigns to the corresponding experts.

### 3. Dispatch and Expert Computation

In actual parallel matrix computation, this is implemented through index masks or selection matrices rather than loops.

#### 3.1 Defining a Selection Matrix

For expert $i$, define a binary selection matrix $M_i \in \{0,1\}^{C_i \times T}$, where $C_i$ is the total number of tokens assigned to expert $i$ (its capacity). If input token $t$ is assigned to expert $i$, the corresponding row and column entry in $M_i$ is 1.

#### 3.2 Dispatch through Expert-Input Projection

Matrix multiplication gathers tokens scattered across the batch into expert $i$'s input buffer $X_i \in \mathbb{R}^{C_i \times d}$:

$$
X_i = M_iX
$$

Entry $(m,k)$ of $X_i$ represents dimension $k$ of token $m$ in expert $i$'s input (the embedding space is $d$-dimensional). It is the dot product of row $m$ of $M_i$ and column $k$ of $X$. If expert $i$'s $m$-th input token is token $n$ in the overall sequence, that row of $M_i$ contains $1$ in column $n$ and $0$ elsewhere; column $k$ of $X$ contains dimension $k$ of every token. This effectively “extracts this expert's input matrix from the complete input matrix,” omitting content assigned elsewhere.

#### 3.3 Expert Processing

Expert $i$ processes its assigned input to obtain $Z_i \in \mathbb{R}^{C_i \times d}$:

$$
Z_i = E_i(X_i)
$$

## II. Aggregating Expert Outputs

### 4. Weighted Combination

The final step returns each expert's result $Z_i$ to its original positions and multiplies it by the gating weights.

#### 4.1 Vectorizing Weights

For expert $i$, its gating-weight vector $g_i \in \mathbb{R}^{C_i}$ collects all nonzero values in column $i$ of $G$:

$$
g_i = M_iG_{:,i}
$$

The MoE output $Y \in \mathbb{R}^{T \times d}$ sums all expert results after $M_i^T$ scatters them back to their original positions:

$$
Y = \sum_{i=1}^{N} M_i^T(\mathrm{diag}(g_i) \cdot Z_i)
$$

From the perspective of one token $x$, this is equivalent to the more intuitive sum:

$$
y = \sum_{i \in \mathcal{T}} G(x)_iE_i(x)
$$

Here, $\mathcal{T}$ is the set of selected top-k expert indices.

Detailed explanation:

1. $g_i=M_iG_{:,i}$: a preprocessing step that “aggregates” each expert's weight vector to match the dimensions of subsequent matrix multiplication.

$M_i$ is the binary selection matrix described above, with shape $C_i\times T$ ($C_i$ is the number of tokens selecting expert $i$). It is a sparse binary matrix containing only $0$ and $1$, with one $1$ per row, indicating “which original batch row contains my $k$-th selected token.”

$G_{:,i}$ denotes column $i$ of $G$, a column vector of length $T$ (the total token count). It contains every token's score (weight) for expert $i$ and is sparse, with many $0$ entries. After multiplication, its length becomes $C_i$ (the tokens selecting expert $i$), effectively removing all $0$ entries and concatenating what remains.

The main purpose is to align weight-vector dimensions for subsequent matrix multiplication ($Z_i$ is a $C_i\times d$ matrix).

For example:

Suppose batch size $T=4$ and expert $i=1$. Tokens 1 and 3 select expert 1; tokens 2 and 4 do not.

1. The global weight column $G_{:,1}$ records every token's weight for expert 1:

$$
G_{:,1} =
\begin{pmatrix}
0.9 \\
0 \\
0.8 \\
0
\end{pmatrix}
$$

Token 1 has weight 0.9 (selected), token 2 has weight 0 (not selected), token 3 has weight 0.8 (selected), and token 4 has weight 0 (not selected).

2. Selection matrix $M_1$ indicates that expert 1 handles only tokens 1 and 3, so $M_1$ is a $2 \times 4$ matrix:

$$
M_1 =
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
$$

- Row 1, $[1,0,0,0]$: extract original element 1.
- Row 2, $[0,0,1,0]$: extract original element 3.

3. Compute $g_1$ through $M_1 \times G_{:,1}$:

$$
g_1 =
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
\cdot
\begin{pmatrix}
0.9 \\
0 \\
0.8 \\
0
\end{pmatrix}
=
\begin{pmatrix}
0.9 \\
0.8
\end{pmatrix}
$$

It can also be viewed as a filtering process:

1. $G_{:,i}$ holds a long list describing the attention that 1,000 batch tokens give expert $i$.
2. $M_i$ holds an “admission list” containing only the IDs of the 50 tokens actually entering expert $i$'s room.
3. Multiplication: $M_i$ extracts those 50 attention weights from the long list $G_{:,i}$ and arranges them in order into a shorter vector $g_i$.

2. $\mathrm{diag}(g_i)Z_i$:

Interpretation: after an expert processes data, its results cannot simply be used directly. They must be scaled according to the gating router's “trust.” If the router scores expert $i$ highly (such as $0.9$), most of $Z_i$'s features are retained; with a low score (such as $0.1$), the features are weakened.

Mathematical operation:

- $g_i$ is a vector of length $C_i$.
- $\mathrm{diag}(g_i)$ turns it into a $C_i \times C_i$ diagonal matrix.
- Multiplication by $Z_i$ ($C_i \times d$) effectively performs row-wise scalar multiplication through broadcasting: the output vector of sample $j$ is multiplied by weight scalar $j$.

3. Multiply by $M_i^T$: scatter the rows from this expert's token count back to the full sequence's token count so the outputs can subsequently be added.

Suppose there are four tokens ($T=4$), and expert $E_1$ handles tokens 1 and 3 ($C_1=2$).

Gather stage ($X_i=M_iX$): $M_i$ is a $2 \times 4$ matrix that extracts rows 1 and 3 into a compact $2 \times d$ matrix for the expert.

$$
M_i =
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
$$

Scatter stage ($M_i^T$): the expert has now produced $Z_i'$ (the weighted $2 \times d$ matrix), which must be restored to the original four-row structure. $M_i^T$, the transpose of $M_i$, has shape $4 \times 2$:

$$
M_i^T =
\begin{pmatrix}
1 & 0 \\
0 & 0 \\
0 & 1 \\
0 & 0
\end{pmatrix}
$$

Multiplying $M_i^T$ by the $2 \times d$ result gives:

$$
\begin{pmatrix}
1 & 0 \\
0 & 0 \\
0 & 1 \\
0 & 0
\end{pmatrix}
\cdot
\begin{pmatrix}
\mathrm{Row}_1 \\
\mathrm{Row}_3
\end{pmatrix}
=
\begin{pmatrix}
\mathrm{Row}_1 \\
0 \\
\mathrm{Row}_3 \\
0
\end{pmatrix}
$$

4. Summation: add the experts' output matrices to obtain the final output.

## III. Load-Balancing Loss

To prevent a “Matthew effect,” in which a small number of experts process all data, an auxiliary term is added to the training objective.

Define two probability-distribution vectors:

1. Importance vector ($\Psi$): the cumulative probability mass predicted by the gating network—how the model “wants” to allocate tokens.

$$
\Psi_i = \frac{1}{T}\sum_{t=1}^{T}\mathrm{Softmax}(h_t)_i
$$

2. Load vector ($\Phi$): the actual discrete assignment decisions—how tokens are really allocated.

$$
\Phi_i = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}\left(i \in \mathrm{Indices}(\mathrm{TopK}(h_t,k))\right)
$$

3. Loss: minimize the dot product of these vectors to encourage uniformity.

$$
\mathcal{L}_{aux} = N\sum_{i=1}^{N}\Psi_i \cdot \Phi_i
$$

- The factor $N$ normalizes the loss to 1 under the ideal uniform distribution.
- Total loss: $\mathcal{L}_{total}=\mathcal{L}_{task}+\alpha\mathcal{L}_{aux}$.

A dynamic bias can also be introduced into routing-score calculation:

$$
\mathrm{RoutingScore}=\mathrm{Softmax}(W_{gate}\cdot x+b_{dynamic})
$$

## IV. DeepSeek's Innovations in MoE

DeepSeek did not invent MoE, but introduced important incremental architectural innovations that greatly improved its efficiency.

1. Fine-grained experts

Traditional MoE uses a few large experts (for example, 8). DeepSeek divides them into smaller pieces (for example, 64 smaller experts), activating more small experts per computation.

Mathematical intuition: finer granularity increases flexibility in combining knowledge.

2. Shared-expert isolation

This is DeepSeek's most central contribution.

Traditional MoE's problem: some knowledge, such as grammatical structure and basic logic, is general. Traditional MoE forces this knowledge to be stored redundantly in every expert, wasting parameters.

DeepSeek's approach changes the output formula to:

$$
y=\sum_{i\in \mathrm{TopK}}G_i(x)E_i(x)+E_{shared}(x)
$$

- $E_{shared}(x)$: an always-active expert dedicated to general knowledge.
- $E_i(x)$: routed experts store only knowledge unique to specialized domains.

## V. The Nondifferentiability of Top-K and Its Solution

1. The affected component: the router

The router (parameters $W_r$) makes the “multiple-choice” decision: given input $x$, which $K$ experts should be selected from hundreds?

Why is it affected?

Top-K is essentially a step function.

- Input: routing logits.
- Output: a few discrete indices, such as `{Expert 5, Expert 9}`.

Problem scenario: suppose the loss is high, and gradient descent should tell the router, “Your selection was wrong; you should have chosen Expert 7 instead of Expert 5!”

Mathematically, however, Top-K produces zero gradients:

1. A tiny change to routing parameters $W_r$ slightly changes the scores (for example, Expert 5 goes from 0.8 to 0.801 and Expert 7 from 0.79 to 0.791).
2. This is insufficient to change the top-K ranking, which still selects Expert 5.
3. Since the result does not change, neither does the loss.
4. Conclusion: the derivative of the loss with respect to $W_r$ is 0.

Consequence: without a remedy such as DeepSeek's weighted sum $g_i \cdot E_i$, the router cannot learn. It remains at its random initialization because it receives no feedback that changing parameters can reduce loss.

2. The unaffected components: the experts

Expert networks (parameters $W_e$) perform the actual work.

Why are they unaffected?

- An expert does not care which decision logic selected it.
- Once selected, it receives data.
- Once it receives data, its computed result contributes to the final loss.
- Once it contributes to the computation, gradients flow backward along that data path.

For an expert, Top-K is merely a mask acting as a switch.

- Switch on (selected): train normally and update parameters.
- Switch off (not selected): remain idle with unchanged parameters.

Top-K's nondifferentiability does not block “loss → expert output → expert weights”; it only blocks “loss → selection logic → routing weights.”

3. How does DeepSeek “revive” the router?

Since Top-K prevents learning through changing the selection, DeepSeek and modern MoE let the router learn by **changing the magnitude**.

Recall:

$$
y=\sum_{i\in T}g_i\cdot E_i(x)
$$

The router's gradient-backpropagation path becomes:

$$
\mathrm{Loss}\rightarrow y\rightarrow g_i\;(\mathrm{Softmax\ Probability})\rightarrow \mathrm{Logits}\rightarrow W_r
$$

The key is that the model cannot directly tell the router to “replace Expert 5 with Expert 7” (a discrete jump), but can tell it:

“Expert 5 did well; increase its weight $g_5$ to make it more important,” or “Expert 9 did poorly; decrease its weight $g_9$.”

Through continuous adjustment of $g_i$, if an unselected expert's latent score (such as Expert 7's) gradually rises past the top-K threshold, it will suddenly be selected at some point. In this indirect way, the router uses continuous weight gradients to optimize discrete choices.

## References

- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). [Adaptive Mixtures of Local Experts](https://doi.org/10.1162/neco.1991.3.1.79). Neural Computation.
- Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538). arXiv:1701.06538.
- Dai, D., Deng, C., Zhao, C., et al. (2024). [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066). arXiv:2401.06066.
- DeepSeek-AI. (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). arXiv:2412.19437.
- Gemini Team, Google. (2024). [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2403.05530). arXiv:2403.05530.
