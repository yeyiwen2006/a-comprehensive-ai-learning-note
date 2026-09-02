---
title: "19.8 线性注意力（Linear Attention）"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.8 线性注意力（Linear Attention）.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.8 线性注意力（Linear Attention）

## 一、基本原理

线性化注意力的核心思想是：通过改变计算顺序，避免显式地计算和存储这个 $n \times n$ 的注意力矩阵 $A$。

它的目标是将计算复杂度从 $O(n^2)$ 降低到 $O(n)$。

如何实现？答案是利用矩阵乘法的结合律。

- 标准计算顺序：$(QK^T)V$，先算 $n \times n$ 矩阵，再与 $V$（$n \times d$）相乘。
- 线性化计算顺序：$Q(K^T V)$，先算 $K^T V$（得到一个 $d \times d$ 的矩阵），再与 $Q$（$n \times d$）相乘。

实现线性化并非无条件的，必须去掉 Softmax，这是因为：

标准注意力（Standard Attention）为：

$$
\mathrm{Output} = \mathrm{Softmax}(QK^T)V
$$

请注意，这里不仅有乘法，还有一个非线性的 Softmax 包裹在中间。

$$
\mathrm{Softmax}(QK^T) \ne Q \cdot \mathrm{Softmax}(K^T)
$$

你不能把 Softmax 拆开，也不能把它挪到后面去先算 $K^T V$。因为 $Q$ 和 $K$ 的每一项都必须先交互、求和、归一化之后，才能和 $V$ 相乘。这意味着你必须构建那个巨大的 $n \times n$ 矩阵。

我们需要一个核函数来近似 Softmax 函数。具体数学推导如下。

首先，我们把标准注意力写成一个按行归一化的形式。对于第 $i$ 个查询向量 $q_i$ 的输出 $o_i$：

$$
o_i =
\frac{
\sum_{j=1}^{n} \exp\left(\frac{q_i k_j^T}{\sqrt{d_k}}\right)v_j
}{
\sum_{j=1}^{n} \exp\left(\frac{q_i k_j^T}{\sqrt{d_k}}\right)
}
$$

我们可以将其抽象为一个更通用的相似度函数 $\mathrm{sim}(\cdot,\cdot)$：

$$
o_i =
\frac{
\sum_{j=1}^{n} \mathrm{sim}(q_i,k_j)v_j
}{
\sum_{j=1}^{n} \mathrm{sim}(q_i,k_j)
}
$$

接着引入核函数 $\phi$，用特征映射后的内积来表示相似度：

$$
\mathrm{sim}(q_i,k_j) = \phi(q_i)\cdot\phi(k_j)^T
$$

其中 $\phi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d_{proj}}$ 是一个将原始向量映射到另一个特征空间的函数，它也称为核函数。

现在，把这个核函数代入输出 $o_i$ 的计算中：

$$
o_i =
\frac{
\sum_{j=1}^{n}[\phi(q_i)\cdot\phi(k_j)^T]v_j
}{
\sum_{j=1}^{n}[\phi(q_i)\cdot\phi(k_j)^T]
}
$$

由于 $\phi(q_i)$ 对于求和索引 $j$ 是常数，我们可以将其提到求和号外面：

$$
o_i =
\frac{
\phi(q_i)\cdot\left[\sum_{j=1}^{n}\phi(k_j)^T \otimes v_j\right]
}{
\phi(q_i)\cdot\left[\sum_{j=1}^{n}\phi(k_j)^T\right]
}
$$

这里 $\otimes$ 表示外积；在固定行、列向量方向后，也常写成矩阵乘法。

注：内积表示行向量乘列向量，结果为标量；外积表示列向量（$m \times 1$）乘行向量（$1 \times n$），结果为矩阵（$m \times n$）。

让我们定义两个关键的累积状态矩阵：

1. $S$：键和值的聚合状态。

$$
S = \sum_{j=1}^{n}\phi(k_j)^T v_j \in \mathbb{R}^{d_{proj}\times d_v}
$$

注意：这里的 $\phi(k_j)$ 是行向量，它转置后是列向量，与 $v_j$（行向量）做外积，得到一个矩阵。我们对所有 $j$ 的这样的矩阵求和。

2. $Z$：归一化因子的聚合状态。

$$
Z = \sum_{j=1}^{n}\phi(k_j)^T \in \mathbb{R}^{d_{proj}\times 1}
$$

现在，对于整个序列的输出 $O$ 中的第 $i$ 行 $o_i$，可以计算为：

$$
o_i = \frac{\phi(q_i)S}{\phi(q_i)Z}
$$

这里的分母 $\phi(q_i)Z$ 是一个标量（归一化因子），因此输出向量的每个分量都除以同一个归一化值。

- 计算 $S$ 和 $Z$：遍历序列中的每个位置 $j$，求 $\phi(k_j)^T v_j$ 和 $\phi(k_j)^T$，然后累加。这一步的复杂度是 $O(n\cdot d_{proj}\cdot d_v)$。
- 计算所有输出 $o_i$：对于每个位置 $i$，计算 $\phi(q_i)S$ 和 $\phi(q_i)Z$。这一步的复杂度也是 $O(n\cdot d_{proj}\cdot d_v)$。

总复杂度是 $O(nd_{proj}d_v)$。如果我们固定 $d_{proj}$ 和 $d_v$，它们是与序列长度 $n$ 无关的常数，那么复杂度就是 $O(n)$，即线性复杂度。

注：$K$ 和 $V$ 本来分别为 $n \times d_k$ 和 $n \times d_v$ 维，$K$ 转置后与 $V$ 矩阵相乘，矩阵乘法计算复杂度为 $n\cdot d_k\cdot d_v$，也就是相对于序列长度的 $O(n)$。这里的 $\phi$ 只改变特征维度，不改变对序列位置逐项累加的线性结构。

## 二、如何寻找核函数 $\phi(x)$？

对于 Softmax 中的指数相似度，我们希望找到某种特征映射，使得：

$$
\mathrm{sim}(q,k) = \exp(q\cdot k^T) \approx \phi(q)\cdot\phi(k)^T
$$

在深度学习模型（如 Transformer 的原始注意力机制）中直接使用 Softmax 时，我们是在进行数值计算。对于单个数值 $x$，$\exp(x)$ 是一个成熟的数学函数，计算机可以高效处理。但当 $q$ 和 $k$ 是两个向量时，若希望把 $\exp(q\cdot k)$ 写成两个特征向量的内积，就会自然引出更高维的特征映射。

从泰勒展开开始：

$$
\exp(q\cdot k)
= 1 + (q\cdot k) + \frac{(q\cdot k)^2}{2!}
+ \frac{(q\cdot k)^3}{3!}
+ \frac{(q\cdot k)^4}{4!}
+ \cdots
$$

对于 $m=1$：

$$
(q\cdot k)^1 = \sum_{i=1}^{d}q_i k_i
$$

这可以看作是向量 $q$ 和 $k$ 在原始 $d$ 维空间中的内积。

对于 $m=2$：

$$
(q\cdot k)^2
= \left(\sum_{i=1}^{d}q_i k_i\right)^2
= \sum_{i=1}^{d}\sum_{j=1}^{d}q_i q_j k_i k_j
$$

注意，这可以重写为两个新向量的内积：

- 令 $\phi_2(q)$ 是一个 $d^2$ 维的向量，包含所有 $q_i q_j$ 的组合。
- 令 $\phi_2(k)$ 是一个 $d^2$ 维的向量，包含所有 $k_i k_j$ 的组合。

那么：

$$
(q\cdot k)^2 = \phi_2(q)\cdot\phi_2(k)
$$

现在，我们可以为每个幂次 $m$ 定义一个特征映射：

- 对于 $m=0$：$\phi_0(q)=1$，是 1 维。
- 对于 $m=1$：$\phi_1(q)=q$，是 $d$ 维。
- 对于 $m=2$：$\phi_2(q)$ 包含所有 $q_i q_j$，是 $d^2$ 维。
- 对于 $m=3$：$\phi_3(q)$ 包含所有 $q_i q_j q_k$，是 $d^3$ 维。
- 依此类推。

将这些不同阶数的特征组合起来，就可以构造一个无限维的超级特征映射 $\Phi(q)$：

$$
\Phi(q)=
\left(
1,\;
q,\;
\frac{q_i q_j}{\sqrt{2!}},\;
\frac{q_i q_j q_k}{\sqrt{3!}},\;
\frac{q_i q_j q_k q_l}{\sqrt{4!}},\;
\cdots
\right)
$$

这里每个分量都除以相应的 $\sqrt{m!}$，用于匹配泰勒展开中的系数。

于是：

$$
\begin{aligned}
\Phi(q)\cdot\Phi(k)
&= 1\cdot 1
+ \sum_i q_i k_i
+ \sum_{i,j}\frac{q_i q_j}{\sqrt{2!}}\frac{k_i k_j}{\sqrt{2!}}
+ \sum_{i,j,k}\frac{q_i q_j q_k}{\sqrt{3!}}\frac{k_i k_j k_k}{\sqrt{3!}}
+ \cdots \\
&= 1 + \sum_i q_i k_i
+ \frac{1}{2!}\sum_{i,j}q_i q_j k_i k_j
+ \frac{1}{3!}\sum_{i,j,k}q_i q_j q_k k_i k_j k_k
+ \cdots
\end{aligned}
$$

但注意：

$$
\sum_i q_i k_i = q\cdot k
$$

$$
\sum_{i,j}q_i q_j k_i k_j = (q\cdot k)^2
$$

$$
\sum_{i,j,k}q_i q_j q_k k_i k_j k_k = (q\cdot k)^3
$$

因此：

$$
\Phi(q)\cdot\Phi(k)
= 1 + (q\cdot k) + \frac{(q\cdot k)^2}{2!}
+ \frac{(q\cdot k)^3}{3!}
+ \cdots
= \exp(q\cdot k)
$$

计算 $\exp(q\cdot k)$ 这个看似简单的操作，在数学上等价于：将向量 $q$ 和 $k$ 映射到一个无限维的特征空间，在这个无限维空间中计算它们的内积。这个无限维空间包含了原始向量所有可能的多项式组合，包括一次项、二次项、三次项等。

线性注意力的出现本身就是为了解决长序列上 $O(n^2)$ 的问题，但如果直接使用这个无限维特征映射，它自身又会带来趋于无限维的特征空间。在一个本身就可能趋于无限长的序列上，这是计算机无法处理的。

因此，我们必须对这个函数进行近似。常见近似方法如下。

1. $\phi(x)=\mathrm{elu}(x)+1$。

   ELU 函数：对于 $x>0$，输出 $x$；对于 $x\le 0$，输出 $\alpha(\exp(x)-1)$，通常 $\alpha=1$。

   $\mathrm{elu}(x)+1$：对于 $x>0$，约为 $x+1$（线性增长）；对于 $x\le 0$，变为 $\exp(x)$（指数增长）。

   设计动机：这个函数在 $x\le 0$ 时具有类似指数函数的行为，而在 $x>0$ 时保持线性，整体上试图模拟指数函数的形状。但它是有限维的，输出维度与输入 $x$ 相同。

2. $\phi(x)=\mathrm{ReLU}(x)^2$。

   这是一个简单的多项式函数。$\mathrm{ReLU}(x)$ 将负值置零、正值保留，然后再平方。

   它对应一个二阶多项式核，计算简单，但近似能力相对较弱。

3. 基于随机特征的方法。

   这是一种数学上更严谨的近似方法。它通过随机投影构造一个显式、有限维的特征映射 $\phi(x)$，使得 $\phi(q)\cdot\phi(k)$ 近似于某个目标核函数。

4. 学习得到的核函数。

   不手动设计 $\phi$，而是让模型自己学习。可以用一个小型神经网络（如一层 MLP）作为 $\phi$，在训练过程中通过梯度下降优化它，使其能更好地配合整个模型完成任务。

但这些核函数都不能足够好地拟合 Softmax，这是线性注意力性能不如 Full Attention 的核心原因。

## 三、在自回归中的优势

线性化注意力还有一个独特优势：它可以被轻松地改写为递归形式，非常适合自回归推理。

令：

$$
S_t = \sum_{j=1}^{t}\phi(k_j)^T v_j
= S_{t-1}+\phi(k_t)^T v_t
$$

$$
Z_t = \sum_{j=1}^{t}\phi(k_j)^T
= Z_{t-1}+\phi(k_t)^T
$$

在生成第 $t$ 个 token 时：

1. 我们已经有之前所有 token 的累积状态 $S_{t-1}$ 和 $Z_{t-1}$。
2. 我们计算当前 token 的 $k_t$、$v_t$，并更新状态：

$$
S_t = S_{t-1}+\phi(k_t)^T v_t,\quad
Z_t = Z_{t-1}+\phi(k_t)^T
$$

3. 然后使用当前查询 $q_t$ 计算输出：

$$
o_t = \frac{\phi(q_t)S_t}{\phi(q_t)Z_t}
$$

这就像一个 RNN。在生成第 $t$ 个 token 时，我们已经有了前 $t-1$ 个 token 的完整总结 $S_{t-1}$ 和 $Z_{t-1}$。当新的第 $t$ 个 token 产生时，我们不需要回溯整个历史，只需要将当前 token 的贡献 $\phi(k_t)^T v_t$ 加到 $S_{t-1}$ 上，将 $\phi(k_t)^T$ 加到 $Z_{t-1}$ 上，就得到了新的状态 $S_t$ 和 $Z_t$。然后用当前查询 $q_t$ 与最新状态交互，得到当前时刻的输出。

因此，我们不需要存储整个键值的历史，只需要维护一个固定大小的状态 $S$ 和 $Z$。每一步的计算成本是常数 $O(1)$（相对于序列长度），极大地提高了长文本生成的效率。

## 参考文献

- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). ICML 2020.
- Shen, Z., Zhang, M., Zhao, H., Yi, S., & Li, H. (2021). [Efficient Attention: Attention with Linear Complexities](https://arxiv.org/abs/1812.01243). WACV 2021.
