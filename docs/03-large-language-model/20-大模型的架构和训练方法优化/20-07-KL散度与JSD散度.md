---
title: "20.7 KL散度与JSD散度"
source_docx: "第3部分 大语言模型/20.大模型的架构和训练方法优化/20.7 KL散度与JSD散度.docx"
status: "image-reconstructed"
ocr: "manual reconstruction completed from classified DOCX images"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 20.7 KL散度与JSD散度


在大模型中，KL散度衡量输出概率分布的差异，并非按输出前的隐藏层维度计算，而是在大小为V的词表上计算（因为本质上大模型的输出是离散的tokens）。

D(P||Q)=sigma(pi*log(pi/qi)),i=1,2,...,V

注意：计算困惑度时是对序列中的所有token的概率负对数取平均，而计算KL散度时是在输出一个token时，对词表中的所有词的KL散度取平均，二者取平均的方式不同。

实际研究中，为了让数值更稳定，往往用KL散度的变体——JSD散度，即取P和Q的平均M，JSD散度=1/2*(KL(P||M)+KL(Q||M))，功能和KL类似但不会出现log内分母为零，例如研究模型中间隐藏层和输出层之间神经元激活值的JSD散度（如果二者维度不同，则投影后再用JSD散度），就可以判断二者分布之间的相似程度大小。

对于LLM，自回归输出n个token的KL散度：

$$
D_{\mathrm{KL}}(P(Y)\Vert Q(Y))
=
\sum_Y P(Y)\log\frac{P(Y)}{Q(Y)}.
$$

由于直接对所有可能的序列求和是不可行的（有V^n种可能），需利用自回归特性将其分解：

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

这里的处理难点在于“Y~P”，需要考虑整个序列的概率分布。注意到右边的条件概率的条件为y<t，对于给定已知的y<t分布，由自回归特性，yt的分布也会随之确定，Y={y1,...,yt}的分布也就已知。因此可以改写为：

$$
D_{\mathrm{KL}}(P(Y)\Vert Q(Y))
=
\sum_{t=1}^{n}
\mathbb{E}_{y_{<t}\sim P(y_{<t})}
\left[
D_{\mathrm{KL}}\left(P(y_t\mid y_{<t})\Vert Q(y_t\mid y_{<t})\right)
\right].
$$

这意味着，两个序列分布的KL散度，等于每一步条件概率分布之间KL散度的期望之和（期望是基于分布P生成的历史轨迹计算的）。

## 参考文献

暂无已核验参考文献。
