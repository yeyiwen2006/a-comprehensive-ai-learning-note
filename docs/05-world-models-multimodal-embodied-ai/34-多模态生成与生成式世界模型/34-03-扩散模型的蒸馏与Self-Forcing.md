---
title: "34.3 扩散模型的蒸馏与Self Forcing"
source_docx: "第5部分 世界模型、多模态生成与具身智能/34.多模态生成与生成式世界模型/34.3 扩散模型的蒸馏与Self Forcing.docx"
status: "auto-converted"
ocr: "image placeholders rebuilt as Markdown/LaTeX"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 34.3 扩散模型的蒸馏与Self Forcing

视频扩散模型虽然视觉保真度极高，但由于需要迭代去噪全部帧，生成一段短视频通常需要1到2分钟，这在实时人机交互和机器人领域并不现实。通过改进的蒸馏技术，可将庞大缓慢的模型转化为极速的实时模型。同时，视频生成模型仅以Ground Truth为上文进行Next State Prediction极易导致训练崩溃，需要自主适应长轨迹生成，具备在过程中纠错的能力。

## 一、一致性蒸馏

一致性模型（Consistency Models, CM）的原始思想是：无论在概率流 ODE 轨迹上的哪一个时间点 $t$，我们都训练一个学生网络 $f_\theta(x_t,t)$，让它直接一步映射回轨迹的终点（即干净数据 $x_0$）。其数学约束为：

$$
f_\theta(x_t,t)=f_\theta(x_{t'},t')
$$

这种方法在训练中一般用于模型轨迹初始化，让学生模型直接利用教师模型生成的轨迹训练，快速学会去噪。由于教师模型往往每生成一帧都需要走较多的步数（如 $N=48$ 步），但为了降低延迟，学生模型并不能走这么多步，故只采样关键时间步。由于教师模型和学生模型走的步数不同，这里改为预测真实数据，而不是教师预测的噪声。

为了让模型具备快速生成和自回归（逐块生成）的能力，第一步需要为学生模型打下坚实的基础，让它学会用极少的步数完成原本需要很多步的去噪过程。

- 工作流：
  1. 系统首先运行庞大的教师模型，生成完整的 $N=48$ 步去噪轨迹数据 $\{x_{t_j}\}_{j=0}^{N}$。
  2. 算法从这条极长的轨迹中，极其精简地子采样出 $k=4$ 个关键时间步。
  3. 因果学生生成器 $g_\phi$ 接收带噪潜变量和多模态条件 $c$（文本、音频、参考图像），并尝试在这 4 步内直接预测出没有任何噪声的纯净视频块 $x_0$。
- 数学公式：

在此阶段，学生模型通过最小化预测值与真实无噪数据之间的均方误差进行学习。

$$
\mathcal{L}_{ODE}=\mathbb{E}\left[\left\|g_\phi(x_{t_j},c)-x_0\right\|_2^2\right]
$$

## 二、轨迹分段一致性蒸馏（TSCD）

直接让网络跨越极长的时间步去预测 $x_0$，对于视频这种高维且高度非线性的数据来说，学习难度极大，容易陷入局部最优并导致画面模糊。引入自 HyperSD 的 TSCD 技术，其核心思想是“分而治之”。

1. 时间域分段：将完整的时间轨迹 $[0,T]$ 划分为 $k$ 个连续的子区间：

$$
[t_0,t_1],[t_1,t_2],\ldots,[t_{k-1},t_k],\quad t_0=0,\ t_k=T
$$

2. 分段一致性约束：在每一个子区间 $[t_{i-1},t_i]$ 内，学生网络不再被强迫去预测遥远的 $x_0$，而是只被要求保持该区间内的一致性。在生成（去噪）时，如果模型当前处于时间点 $t_i$，它不再好高骛远地去预测遥远的 $x_0$，而是只预测当前分段的下界端点，即更接近真实数据的状态 $x_{t_{i-1}}$。

3. 优势：通过将一条极长且曲折的积分曲线切分成多段相对平滑的短曲线，学生模型拟合扩散过程的难度大幅降低。在推理时，模型只需在每个预设的宏观节点跳跃，就能极其精确地逼近完整的扩散轨迹，从而在较少步数下提供速度与保真度的强劲平衡，实现 4 倍加速。

## 三、Self forcing与分布匹配蒸馏（DMD）

### （一）核心思想

基于 Teacher Forcing 的一致性蒸馏是一个 Off-Policy 的过程。由于学生模型的生成在块间是自回归的，一旦学生模型的生成轨迹发生偏离，就会进入 OOD 区域，这就意味着模型极不稳定。我们希望进行 On-Policy 蒸馏，即 Self Forcing：在学生模型已有生成的基础上，让学生生成的最终数据分布 $p_\Phi(x)$ 接近真实的物理世界数据分布 $p_{\mathrm{data}}(x)$（可以由教师模型近似代替）。

研究表明，在Self forcing中如果直接对学生生成的数据和真实数据用MSE Loss，极易导致训练崩溃，但用分布匹配蒸馏方法能解决这个问题。后续会解释原因。

在概率统计中，衡量两个分布差异最经典的工具是 Kullback-Leibler（KL）散度。我们希望最小化 $\mathcal{D}_{\mathrm{KL}}(p_\phi\|p_{\mathrm{data}})$。

重点来了：在 Score-based 扩散模型的数学框架下，要通过梯度下降来缩小这两个分布的差距，其损失函数对生成数据的梯度，恰好等于这两个分布的“分数”（Score）之差：

$$
\begin{aligned}
\nabla_x\mathcal{D}_{\mathrm{KL}}(p_\phi\|p_{\mathrm{data}})
&=\nabla_x\log p_\phi(x)-\nabla_x\log p_{\mathrm{data}}(x)
\end{aligned}
$$

第二项是真实去噪数据分布对数的梯度，根据扩散模型的基本原理，教师模型学习真实数据分布，故这一项约等于教师模型输出的噪声预测；第一项是学生模型预测的去噪数据分布对数的梯度，我们用一个“评论家网络”学习学生模型输出数据分布，在其上进行加噪，噪声预测即为本项，从而计算梯度，更新学生模型的参数。

### （二）工作流

1.评论家网络学习学生模型的去噪方式

评论家网络的工作流如下：

1. 学生生成器 $g_\phi$ 根据高斯噪声 $z$ 和多模态条件 $c$ 生成预测视频帧 $\hat{x}_0$，即 $\hat{x}_0=g_\phi(z,0,c)$。
2. 系统给这个生成的 $\hat{x}_0$ 重新人为添加 $\tau$ 时刻的噪声，得到 $x_\tau$。
3. 评论家网络 $s_\psi$ 的任务是充当“裁判”，它需要紧紧追踪学生生成器不断变化的生成分布，尝试对 $x_\tau$ 进行去噪，以还原出 $\hat{x}_0$。

数学公式为：

$$
\begin{aligned}
\mathcal{L}_{critic}
&=\mathbb{E}_{\tau}\left[\left\|s_\psi(x_\tau,\tau,c)-\hat{x}_0\right\|_2^2\right]
\end{aligned}
$$

2.根据教师模型与评论家网络的差距，梯度更新学生模型

现在我们有了不断生成的“学生”和一个能追踪学生水平的“裁判”，最后一步就是引入一个冻结的、拥有绝对正确知识的“金牌教师分数网络” $s_\theta$，来强行纠正学生的错误。

注意：多模态条件 $c$ 的加入使这里的计算极易崩溃，因此论文才强调必须搭配高质量的训练数据和激进的学习率调度。

整个梯度更新交替进行的工作流如下：

1. 单步生成：学生生成器 $g_\phi$ 根据噪声和条件直接生成预测数据 $\hat{x}_0$。
2. 随机加噪：系统给这个生成的 $\hat{x}_0$ 重新人为添加 $\tau$ 时刻的噪声，得到加噪数据 $\hat{x}_\tau$（文档文字部分也记作 $x_\tau$）。
3. 计算分数差异：将同一个加噪数据 $\hat{x}_\tau$ 和条件 $c$ 同时喂给“金牌教师”网络 $s_\theta$ 和“裁判”评论家网络 $s_\psi$。
4. 立即反向传播：计算两者给出的分数（梯度）差异，这个差值构成了精准的惩罚信号，并通过链式法则立刻反向传播给学生生成器 $g_\phi$，更新其参数。数学公式表示为：

$$
\begin{aligned}
\nabla_\phi\mathcal{L}_{DMD}
&=\mathbb{E}_{\tau}\left[
\frac{\partial\hat{x}_0}{\partial\phi}
\frac{s_\psi(\hat{x}_\tau,\tau,c)-s_\theta(\hat{x}_\tau,\tau,c)}{\tau}
\right]
\end{aligned}
$$

5. 交替更新：整个 DMD 训练过程就是不断在“更新评论家网络”和“更新学生生成器”之间交替进行。

这个过程是On-policy的，因为是用学生模型的生成进行加噪后，再作为数据给教师模型和评论家网络。

### （三）为什么DMD效果良好，MSE Loss却会崩溃？

MSE 的致命缺陷（均值回归问题）：如果要求 Student 模型只用 1 步就生成图像，它生成的轨迹必然会和多步的 Teacher 模型不同。比如，给定相同的文本，Teacher 画了一只黄色的猫，Student 画了一只黑色的猫。两者都是合理的，但如果在像素级强行计算 MSE，系统会认为 Student 犯了弥天大错，并强迫 Student 去拟合黄色和黑色的平均值，最终模型会生成一团模糊的灰色。

DMD 的解决方案（分布级对齐）：DMD 放弃了像素级的强对齐，转而使用 KL 散度来衡量“生成分布”和“真实分布”的整体差异。其核心梯度近似为：

$$
\begin{aligned}
\nabla_\theta\mathcal{L}_{DMD}
&\approx
\mathbb{E}\left[
w(t)
\left(\hat{\epsilon}_{fake}(\hat{x}_t,t)-\hat{\epsilon}_{real}(\hat{x}_t,t)\right)
\frac{\partial\hat{x}_t}{\partial\theta}
\right]
\end{aligned}
$$

- $\hat{\epsilon}_{real}$：由冻结的强大 Teacher 模型（代表真实数据分布）计算出的分数。
- $\hat{\epsilon}_{fake}$：由一个专门拟合 Student 虚假数据的模型计算出的分数。

DMD 的逻辑是：只要 Student 生成的内容（比如黑猫）符合真实世界的概率分布，Teacher 认为黑猫也很合理，$\hat{\epsilon}_{real}$ 和 $\hat{\epsilon}_{fake}$ 差异很小，就不产生惩罚梯度。

当然，DMD也存在缺点。除算力开销大、需要强大的教师模型外，还需考虑不稳定问题：

MSE 的最大优点是“老实且稳定”。对于任意一个像素点，预测值和真实值的欧氏距离是一个极其平滑的凸函数近似（在局部），这使得损失在极其庞大的集群上进行分布式训练时，能非常稳定地下降，这也是 Scaling Law 的基石。

而分布匹配（无论是基于对抗网络的 GAN Loss，还是基于分数匹配的 DMD）是一个动态博弈的过程。

- 不稳定性：在训练过程中，Student 模型和评估分布差异的 Fake 模型（或判别器）相互拉扯。如果其中一方进化过快，梯度就会瞬间爆炸或消失，导致整个耗资巨大的训练集群停摆。
- 模式崩溃（Mode Collapse）：为了使得全局分布差异最小，模型常常会“投机取巧”。它发现只要反复生成某几种它极其擅长的高质量画面（比如永远只生成慢动作的静止风景），就能在分布评估中获得高分。这会导致模型丧失生成多样性（Diversity）。

### （四）梯度截断

在标准的自回归生成中，如果我们生成一个长度为 $N$ 的视频序列，且每一帧需要经过 $T$ 步扩散去噪（Diffusion Denoising Steps），那么完整的计算图深度将是 $N\times T$。

如果使用标准的随时间反向传播（BPTT, Backpropagation Through Time），我们需要将这 $N\times T$ 步的所有前向激活值（Activations）都保存在显存中，以便反向传播时计算梯度。这种操作会导致显存占用随序列长度和扩散步数呈指数级爆炸，即便是拥有 80GB 显存的顶级算力卡（如 H100）也无法承受几秒钟视频的完整 BPTT。

因此，随机梯度截断通过在计算图上进行“剪枝”，在保证模型能够学习到容错能力的前提下，将显存占用限制在一个恒定的、可接受的范围内。

具体截断方法包括以下两种：

1.沿时间步数截断

这指的是第i步生成中的梯度最多回传到第i-m步，防止计算图深度过大。

2.沿去噪步数截断

即使切断了帧与帧之间的梯度，单帧内部的 $T$ 步去噪（哪怕在使用 Few-step 模型时 $T=4$）依然会占用较多显存。

随机截断策略会在训练的每次迭代中，为序列中的每一帧 $i$ 随机采样一个去噪步数 $s_i\in\{1,2,\ldots,T\}$。在反向传播时，只计算从第 $T$ 步到第 $s_i$ 步的梯度，直接丢弃更早去噪步骤的计算图。这不仅节省了显存，还引入了随机性，起到了类似 Dropout 的正则化效果。

### （五）工程策略

多模态的引入使得DMD训练极其脆弱。可改进如下：

- 优化多模态条件数据：使用视觉大模型（如 Qwen-Image）提升参考图像的质量，并使用 Qwen2.5-VL 强化文本提示词中的动态特征描述，以防止低质量数据引发的误差崩塌。
- 充分收敛的 ODE 初始化：不同于以往文本到视频的蒸馏，本模型必须在 ODE 阶段训练至完全收敛（20k 步），以此为后续敏感的生成器-评论家博弈提供极其稳固的起点。
- 激进的优化调度：鉴于多模态蒸馏的有效学习窗口非常短暂（通常在几百步后开始退化），研究团队将学习率翻倍，并大幅提升教师模型的分类器引导（CFG）比例，从而在窗口期内强制模型学会唇轨同步等高难度对齐任务。

## 四、流匹配模型中的DMD

目标是最小化学生生成分布 $p_{\mathrm{fake}}$ 与真实数据分布 $p_{\mathrm{real}}$ 之间的 KL 散度。其关于生成器参数 $\theta$ 的核心梯度表达式如下：

$$
\begin{aligned}
\nabla_\theta\mathcal{L}_{\mathrm{DMD}}
&=
\mathbb{E}_{z,\epsilon,t}\left[
\omega(t)
\left(v_{\mathrm{teacher}}(x_t,t)-v_{\mathrm{fake}}(x_t,t)\right)
\nabla_\theta G_\theta(z)
\right]
\end{aligned}
$$

其中各变量的含义如下：

- $z\sim\mathcal{N}(0,I)$ 是输入给学生生成器的初始噪声。
- $G_\theta(z)$ 是学生模型生成的图像（即假样本）。
- $\epsilon\sim\mathcal{N}(0,I)$ 是用于构造中间状态的参考噪声。
- $t\in[0,1]$ 是时间步，通常从均匀分布中采样。
- $x_t=tG_\theta(z)+(1-t)\epsilon$ 是按照最优传输（Optimal Transport）路径在时间步 $t$ 插值得到的中间状态。
- $v_{\mathrm{teacher}}(x_t,t)$ 是预训练的教师流匹配模型（如 SD3、Flux 等大型流匹配模型）在该状态下预测的真实向量场，它代表指向真实数据流形的方向。
- $v_{\mathrm{fake}}(x_t,t)$ 是一个伴随训练的假向量场预测器（Fake Vector Field Estimator），它负责拟合当前学生模型生成轨迹的向量场。
- $\omega(t)$ 是依赖于时间的权重函数。

要理解上述表达式，我们需要推导流匹配中的向量场 $v_t(x_t)$ 与边缘得分函数 $\nabla_{x_t}\log p_t(x_t)$ 之间的内在数学联系。

在最优传输条件流匹配（OT-CFM）中，概率路径定义为从噪声到数据的直线插值：

$$
x_t=t x_1+(1-t)x_0
$$

其中 $x_0\sim\mathcal{N}(0,I)$ 代表高斯先验，$x_1\sim p_{data}$ 代表目标数据。给定数据终点 $x_1$ 后，$x_t$ 的条件分布为一个高斯分布：

$$
p_t(x_t|x_1)=\mathcal{N}\left(x_t;t x_1,(1-t)^2I\right)
$$

根据条件高斯分布的性质，边缘得分函数可以表示为条件得分的期望：

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\mathbb{E}_{x_1\sim p(x_1|x_t)}
\left[
\nabla_{x_t}\log p_t(x_t|x_1)
\right]
\end{aligned}
$$

对条件分布求导有：

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t|x_1)
&=
-\frac{x_t-tx_1}{(1-t)^2}
\end{aligned}
$$

代入期望中得到边缘得分：

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\frac{t\mathbb{E}[x_1|x_t]-x_t}{(1-t)^2}
\end{aligned}
$$

在流匹配中，目标向量场 $v_t(x_t)$ 定义为路径导数 $\dot{x}_t=x_1-x_0$ 的条件期望。由于 $x_0=\frac{x_t-tx_1}{1-t}$，可以得到：

$$
\begin{aligned}
\dot{x}_t=x_1-\frac{x_t-tx_1}{1-t}
&=
\frac{x_1-x_t}{1-t}
\end{aligned}
$$

取给定 $x_t$ 的条件期望：

$$
\begin{aligned}
v_t(x_t)=\mathbb{E}[\dot{x}_t|x_t]
&=
\frac{\mathbb{E}[x_1|x_t]-x_t}{1-t}
\end{aligned}
$$

从中可以反解出数据 $x_1$ 的后验期望：

$$
\mathbb{E}[x_1|x_t]=x_t+(1-t)v_t(x_t)
$$

将解出的 $\mathbb{E}[x_1|x_t]$ 代回得分函数公式：

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t)
&=
\frac{t(x_t+(1-t)v_t(x_t))-x_t}{(1-t)^2}
\\
&=
\frac{(t-1)x_t+t(1-t)v_t(x_t)}{(1-t)^2}
\\
&=
\frac{t v_t(x_t)-x_t}{1-t}
\end{aligned}
$$

这样，我们就找到了得分函数和流匹配速度场之间的关系。

设真实数据的得分函数值为 $s_{real}$，学生数据的得分函数值为 $s_{fake}$，我们知道：

$$
\begin{aligned}
\nabla_\theta\mathcal{L}_{DMD}
&\propto
\mathbb{E}\left[
w(t)
\left(s_{\mathrm{real}}(x_t,t)-s_{\mathrm{fake}}(x_t,t)\right)
\frac{\partial x_t}{\partial\theta}
\right]
\end{aligned}
$$

而由前面的推导：

$$
\begin{aligned}
s_{\mathrm{real}}(x_t,t)-s_{\mathrm{fake}}(x_t,t)
&=
\frac{t v_{\mathrm{real}}(x_t,t)-x_t}{1-t}
\\
&\quad-
\frac{t v_{\mathrm{fake}}(x_t,t)-x_t}{1-t}
\\
&=
\frac{t}{1-t}\left(v_{\mathrm{real}}(x_t,t)-v_{\mathrm{fake}}(x_t,t)\right)
\end{aligned}
$$

将此结果代回梯度公式，并将链式法则 $\nabla_\theta x_t=t\nabla_\theta G_\theta(z)$ 展开，便得到了流匹配下的 DMD 核心梯度表达式。由转换带来的系数 $\frac{t}{1-t}$ 和链式法则产生的 $t$ 最终被统一吸收到权重函数 $\omega(t)$ 中。

## 五、Checkpointed Self Forcing

### （一）Self Forcing的问题

传统的 Self Forcing 算法要求学生模型和教师模型拥有相同的上下文长度。但在 Solaris 的设计中，为了让学生模型从长上下文的教师模型中获益，必须在生成学生视频时引入滑动窗口（Sliding-window）。

如果在滑动窗口设置下直接进行反向传播，会遇到严重的显存问题：

- 计算图冗余：每生成一帧，滑动窗口就会向前移动一步，产生一个新的上下文窗口。例如，第 1 步是帧 $1:L_s$，第 2 步是帧 $2:L_s+1$。
- 显存爆炸：深度学习框架（如 Jax）的反向传播需要将所有这些重叠的窗口同时保留在内存中。如果学生上下文长度为 $L_s$，总生成步数为 $L_t$，那么这种冗余会导致内存成本高达 $O(L_t\cdot L_s)$。

### （二）Checkpointed Self Forcing的核心思想

在训练时，放弃每一步用滑动窗口的自回归方式推理，而是并行化、用注意力掩码模拟滑动窗口。但这样就无法保证自回归性，故Checkpointed Self Forcing采用了“先不带梯度进行自回归生成，再带梯度并行训练”的方法。工作流如下：

1.无梯度的自回归展开

这个阶段的目的是模拟推理时的自回归生成过程，收集训练所需的“历史干净上下文”和“当前带噪目标”，但在代码层面完全禁用梯度图的构建，以节省极大的显存。

- 全局初始化：设定教师上下文长度 $L_t$（即本次生成的总帧数）和学生上下文长度 $L_s$（即滑动窗口的大小）。初始化两个空列表：$X_0$（用于存放干净估计帧）和 $X_s$（用于存放带噪过渡帧），并初始化一个空的 KV Cache。
- 随机采样截断步数 $s$：在所有的去噪时间步 $\{t_1,\ldots,t_T\}$ 中，均匀随机抽取一个时间步 $s$。这个 $s$ 就是本轮训练中设定的“停止去噪”的目标时刻，对应前文提到的 $\sigma_{stop}$。
- 逐帧生成循环（Outer Loop）：对于要生成的每一帧 $i$（从 1 到 $L_t$）：
  - 注入纯噪声：为第 $i$ 帧初始化一个纯随机噪声 $x_{t_T}^{i}\sim\mathcal{N}(0,I)$。
  - 内部去噪循环（Inner Loop）：从时间步 $T$ 开始，一步步往回去噪，直到时间步 $s$。
  - 如果当前步正好等于截断步 $s$：说明到达了设定的中间噪声水平。算法会将当前的带噪状态 $x_{t_s}^{i}$ 存入 $X_s$ 列表中。接着，模型 $G_\theta$ 基于这个带噪状态和当前的 KV Cache，预测出对应的干净帧 $\hat{x}_0^i$，并将这个干净帧存入 $X_0$ 列表中。

阶段结果：我们得到了两个长度为 $L_t$ 的序列。一个是干净历史帧序列 $X_0=[\hat{x}_0^1,\ldots,\hat{x}_0^{L_t}]$，另一个是带噪过渡帧序列 $X_s=[x_{t_s}^1,\ldots,x_{t_s}^{L_t}]$。此时显存中没有任何用于反向传播的计算图。

2.数据处理

彻底剥离梯度：对 $X_s$ 和 $X_0$ 再次显式调用 `stop_grad()`，确保它们作为纯净的常数输入进入下一步。

拼接序列：将长度为 $L_t$ 的干净帧序列 $X_0$ 和长度为 $L_t$ 的带噪帧序列 $X_s$ 在序列维度上拼接起来，形成一个新的长序列 $X_{in}=[X_0,X_s]$。此时输入序列的总长度翻倍，变为了 $2L_t$。

3.掩码下的并行重计算与反向传播

接下来开始并行重计算，每个token从之前任意抽取的噪声水平开始去噪一步。为了模拟滑动窗口注意力，构建特殊的掩码，可用四个象限表示如下：

- 右下角（带噪 Queries 看带噪 Keys）：带噪帧 $x_{t_s}^i$ 只能看它自己，不允许看其他的带噪帧，也不允许看历史的带噪帧，呈现出一条对角线。
- 左下角（带噪 Queries 看干净 Keys）：带噪帧 $x_{t_s}^i$ 可以看过去的干净帧 $\hat{x}_0$，但必须严格遵守因果关系和滑动窗口限制，即只能看 $\hat{x}_0^{i-L_s:i-1}$ 的信息，以防发生信息穿越。
- 左上角（干净 Queries 看干净 Keys）：干净帧 $\hat{x}_0$ 按照正常的因果滑动窗口规则，看过去的干净帧。
- 右上角（干净 Queries 看带噪 Keys）：全黑，完全禁止。干净帧绝对不允许看到任何带噪帧的信息，以防发生信息穿越。

这样，显存中只需要保留一份大小为 $O(L_t)$ 的计算图，就可以进行反向传播。

## 六、循环内自蒸馏（ILSD）

Google DeepMind在《ELT: Elastic Looped Transformers for Visual Generation》中提出循环内自蒸馏（ILSD），以自由调节扩散模型的去噪循环次数，让每一个中间输出都是有意义的，不至于在没达到预先设定的最大去噪步数时就输出无意义的噪声，从而适配不同硬件算力（如云端用较多去噪步数生成高清图像，端侧用较少去噪步骤数生成普通图像）：

- 教师路径（Teacher Path）：运行完整的最大循环次数 $L_{max}$，产生最成熟、最高保真度的内部表征。
- 学生路径（Student Path）：在每一次训练迭代中，从均匀分布中随机采样一个中途循环次数 $L_{\mathrm{int}}$（$L_{\min}\le L_{\mathrm{int}}\lt L_{\max}$），并在此时提取输出。

训练的目标不仅是让最终输出逼近真实标签，还要让“中间步骤的学生”去模仿“最终完成的教师”。ILSD 的联合损失函数 $\mathcal{L}^{ILSD}_{\Theta}$ 融合了三项：

- 第一项是教师（$L_{max}$）针对真实数据 $y$ 的目标损失（Ground-Truth Loss）。
- 第二项是学生（$L_{int}$）针对真实数据的目标损失。
- 第三项是蒸馏损失，促使学生拉近与教师输出的距离。$sg$ 意味着教师预测在这部分停止梯度传递（stop-gradient）。
- $\lambda$ 是一个随训练进行从 1 线性衰减到 0 的课程学习权重。这迫使共享参数块学会把复杂的变换压缩到更早的循环步骤中。

## 参考文献

- Salimans, T., & Ho, J. (2022). [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512). ICLR.
- Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). [Consistency Models](https://arxiv.org/abs/2303.01469). ICML.
- Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., & Park, T. (2024). [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828). CVPR.
- Huang, X., Li, Z., He, G., Zhou, M., & Shechtman, E. (2025). [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009). arXiv:2506.08009.
- Savva, G., Michel, O., Lu, D., Waiwitlikhit, S., Meehan, T., Mishra, D., Poddar, S., Lu, J., & Xie, S. (2026). [Solaris: Building a Multiplayer Video World Model in Minecraft](https://arxiv.org/abs/2602.22208). arXiv:2602.22208.
- Goyal, S., Agrawal, S., Anil, G. G., Jain, P., Paul, S., & Kusupati, A. (2026). [ELT: Elastic Looped Transformers for Visual Generation](https://arxiv.org/abs/2604.09168). arXiv:2604.09168.
