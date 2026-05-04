---
title: "19.5 Multi-Head Latent Attention"
source_docx: "第3部分 大语言模型/19.注意力机制的工程优化/19.5 Multi-Head Latent Attention.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 19.5 Multi-Head Latent Attention


## 一、基本思想

DeepSeek提出的MLA（Multi-Head Latent Attention）旨在解决现代GPU显存带宽不足的情况下，KV Cache显存占用过大的问题。它通过低秩矩阵分解和矩阵吸收技术，在推理时极大地压缩了KV Cache的体积，又不会出现MQA、GQA的损失问题。

在线性代数中，矩阵的“秩”代表了矩阵中真正独立的、不相关的信息量。假设我们有一个100*100的矩阵，如果第2行是第1行的2倍，第3行是第1行的3倍……那么虽然它100 行，但实际上只有1行是有用的信息。它的秩就是1。

在标准的 Multi-Head Attention 中，我们有H个头（比如32个）。每个头分开反向传播，模型会分别为每个头学习不同的投影矩阵W_K、W_V。事实上，这32个头学到的 Key 和 Value不是完全独立的，它们之间存在极强的相关性。比如处理单词“苹果”：头1关注它的“颜色（红）”，头2关注它的“形状（圆）”，头3关注它的“类别（水果），这些特征虽然不同，但它们都源自同一个语义核心“苹果”。W_K和W_V矩阵用于特征重组，它们事实上关注的是：头关注的特征向量可以看作原特征向量各个维度怎样的加权，既然特征之间存在强相关性，这些权重向量也就存在强相关性。因而DeepSeek MLA 的核心假设是：多头注意力（MHA）中的 Key-Value 矩阵虽然维度很高，但在数学上是“低秩”（Low-Rank）的。即：大部分参数都是冗余的。

数学上，如果一个大矩阵W_Full是低秩的，它就可以被无损（或极低损耗）地分解为两个低秩矩阵W_down*W_up的乘积，在MLA中这两个矩阵分别被称为降维矩阵和升维矩阵。从而我们可以进行如下MLA操作。

## 二、第一步：生成C_KV

在运用 KV Cache 的标准多头注意力中，我们每次读入一个 token（$x_t$），都会生成 $d_{\mathrm{model}}$ 维（维度较高，如 1024 维）的向量 $k_t$ 和 $v_t$（其中包含了各个头，如第 1-32 维为第 1 个头，以此类推），存入显存：

在标准 MHA 中，当前 token 的 Key 和 Value 由输入向量直接投影得到：

$$
k_t = x_t W_K,\qquad v_t = x_t W_V
$$

MLA 认为，由于不同头的向量相关性较高，这里很多维事实上是“浪费”的，我们只需要用一个共用的降维矩阵 $W_{DKV}$ 将真正有用的信息存入一个极低维度（如 64 维）、所有头共用的向量 $c_{KV,t}$，后续每个头再乘以升维矩阵的不同列（保留不同头的多样性）即可。得到 $c_{KV,t}$ 的表达式：

$$
c_{KV,t} = x_t W_{DKV}
$$

位置信息单独储存：

$$
k_t^R = \mathrm{RoPE}(x_t W_{KR})
$$

## 三、第二步：运用结合律对注意力计算进行优化

原本的注意力分数可以看成 Query 与升维解压后的 Key 做点积：

$$
\mathrm{Score} \propto q_t \cdot (c_{KV} \cdot W_{UK})^T
$$

利用矩阵乘法结合律 $(AB)C=A(BC)$，可以把 Key 侧的升维矩阵 $W_{UK}$ 移动到 Query 侧：

$$
\mathrm{Score} \propto (q_t \cdot W_{UK}^T) \cdot c_{KV}^T
$$

我们先让 Query 去乘固定的参数矩阵 $W_{UK}^T$，得到一个新的、被变换过的 Query，记为 $Q_{\mathrm{absorbed}}$，则对于第 $i$ 个注意力头，当前（第 $t$ 个）Token 和第 $j$ 个 Token 间的注意力权重为：

$$
\mathrm{Score}_{t,j}^{(i)}
=
\frac{Q_{\mathrm{absorbed}}^{(i)} \cdot (c_{KV,j})^T}{\sqrt{d}}
$$

由于 $Q_{\mathrm{absorbed}}$ 是对于当前 Token 而言的，因此这一步只涉及 1 个 Token 的计算，开销极小。接下来乘以 $C_{KV}^T$，我们只需要存储 $t*64$ 维的压缩向量 $C_{KV}$，而永远不需要在显存里复原巨大的 $t*1024$ 维的 K 矩阵。在完成“矩阵吸收”后，这步就变成了 $Q_{\mathrm{absorbed}}^{(i)}$ 对 $C_{KV}^T$ 进行类似 MQA 的操作，但保留了 MHA 中不同头的表达能力。

我们还需要加上解耦位置编码，单独计算 RoPE 部分的分数。从而对于第 $i$ 个注意力头，当前（第 $t$ 个）Token 和第 $j$ 个 Token 间的注意力权重为：

$$
\mathrm{Score}_{t,j}^{(i)}
=
\frac{
\left(q_t^{C,(i)} \cdot (W_{UK}^{(i)})^T\right) \cdot c_{KV,j}
+
q_t^{R,(i)} \cdot (k_j^R)^T
}{\sqrt{d}}
$$

其中，第一项 $\left(q_t^{C,(i)} \cdot (W_{UK}^{(i)})^T\right) \cdot c_{KV,j}$ 是内容分数，用于比较当前 token 与历史 token 的语义内容；第二项 $q_t^{R,(i)} \cdot (k_j^R)^T$ 是 RoPE 位置分数，用于比较二者的相对位置信息。

$R_{lt}^{(i)}$ 是第 $i$ 个头独有的、携带位置信息的 Query 向量，可理解为上式中的 $q_t^{R,(i)}$；$k_j^R$ 通常是各头共享的 RoPE Key 向量。它们做点积后，相当于在标准注意力中保留位置匹配能力，使模型仍然能判断 token 之间的相对距离。

## 四、运用分配律对信息聚合（注意力权重*V）进行优化

设 $c_{KV,j}$ 向量的维度为 $d_c$，升维后每个头的特征维度为 $d_h$。在第 $i$ 个头中，注意力分布、压缩潜变量和 Value 升维矩阵分别为：

$$
P_{t,:}^{(i)} = \mathrm{Softmax}(\mathrm{Score}_{t,:}^{(i)}),\qquad
P_{t,j}^{(i)} =
\frac{\exp(\mathrm{Score}_{t,j}^{(i)})}{\sum_{r=1}^{t}\exp(\mathrm{Score}_{t,r}^{(i)})},\qquad
c_{KV,j}\in \mathbb{R}^{d_c},\qquad
W_{UV}^{(i)}\in \mathbb{R}^{d_c \times d_h}
$$

如果按标准路径先把每个历史 token 的压缩潜变量升维成 Value，再做加权求和，则第 $i$ 个头的输出是：

$$
y_t^{(i)}
=
\sum_{j=1}^{t}
\left(P_{t,j}^{(i)} \cdot (c_{KV,j} \cdot W_{UV}^{(i)})\right)
$$

这种写法对每个历史位置 $j$ 都要执行一次 $c_{KV,j} \cdot W_{UV}^{(i)}$ 的升维计算，然后再乘以标量权重 $P_{t,j}^{(i)}$ 并求和。单个头的主要复杂度约为 $O(t \cdot d_c \cdot d_h)$。

DeepSeek 利用矩阵乘法的分配律，把固定的升维矩阵 $W_{UV}^{(i)}$ 移到求和符号之外：

$$
\begin{aligned}
y_t^{(i)}
&= \sum_{j=1}^{t}\left(P_{t,j}^{(i)} \cdot c_{KV,j} \cdot W_{UV}^{(i)}\right) \\
&= \left(\sum_{j=1}^{t}P_{t,j}^{(i)} \cdot c_{KV,j}\right) \cdot W_{UV}^{(i)}
\end{aligned}
$$

于是计算可以拆成两个阶段。第一阶段先在压缩空间内聚合：

$$
\tilde c_{sum}^{(i)}
=
\sum_{j=1}^{t}\left(P_{t,j}^{(i)} \cdot c_{KV,j}\right)
$$

$\tilde c_{sum}^{(i)}$ 是一个 $d_c$ 维低维向量。虽然 $c_{KV,j}$ 来自共享 KV Cache，但每个头的注意力权重 $P_{t,j}^{(i)}$ 不同，所以每个头聚合出的 $\tilde c_{sum}^{(i)}$ 也不同。这个阶段只做标量乘低维向量再求和，复杂度约为 $O(t \cdot d_c)$。

第二阶段再做延迟升维：

$$
y_t^{(i)} = \tilde c_{sum}^{(i)} \cdot W_{UV}^{(i)}
$$

这一步把聚合后的低维结果映射回第 $i$ 个头的 $d_h$ 维特征空间。由于升维只对聚合后的结果做一次，而不是对 $t$ 个历史 token 分别做 $t$ 次，单个头这部分复杂度约为 $O(d_c \cdot d_h)$。

计算出所有 $H$ 个头的 $y_t^{(i)}$ 后，将它们拼接起来，并通过输出投影矩阵 $W_O$ 得到最终的隐状态输出：

$$
y_t
=
\mathrm{Concat}\left(y_t^{(1)}, y_t^{(2)}, \ldots, y_t^{(H)}\right) \cdot W_O
$$

总体而言，MLA 把“先逐 token 升维再聚合”改成“先在压缩空间聚合再延迟升维”，主要复杂度从 $O(t \cdot d_c \cdot d_h)$ 变为 $O(t \cdot d_c + d_c \cdot d_h)$，当历史长度 $t$ 较大时可近似理解为降低约 $d_h$ 倍。

## 参考文献

- DeepSeek-AI. (2024). [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434). arXiv:2405.04434.
