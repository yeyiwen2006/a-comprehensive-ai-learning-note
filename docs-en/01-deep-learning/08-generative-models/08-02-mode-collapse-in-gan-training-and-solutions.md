---
title: "8.2 Mode Collapse in GAN Training and Solutions"
chapter_title: "Generative Models"
section_id: "08-02"
language: en
source_language: zh
source_docx: "第1部分 深度学习/8.生成模型/8.2 GANs训练的模式崩溃问题与解决方案.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 8.2 Mode Collapse in GAN Training and Solutions

## I. Mode Collapse

Suppose we are training a GAN to generate handwritten digits (such as the MNIST dataset, containing 0 through 9). A successful G (generator) should generate all 10 different, realistic digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).

If $G$ and $D$ have “equal adversarial strength” and “evolve together,” this situation does not occur. In practice, however, discriminator $D$ often learns too quickly early in training and soon identifies $G$'s outputs as fake. Because $G$ generates poorly in every direction, the gradient directions received by $G$ are highly chaotic. As discussed earlier, the sigmoid inside $D$ also causes vanishing gradients as $p \to 0$, so these gradients have relatively small magnitudes.

At this point, $G$ may accidentally discover that its digit “7” is especially realistic (with probability significantly greater than $0$) and easily fools $D$ (the discriminator) every time. Since $G$'s sole objective is to fool $D$, amid these chaotic gradients it takes the direction of generating “7” as the only stable gradient direction, often with a relatively large magnitude. This steers the model toward generating more “7”s, producing more positive feedback and more gradients that further steer it toward “7.” Whatever random noise $z$ it receives, it outputs one (or a small group of) similar-looking “7”s. In this situation, we say that $G$ has “collapsed” onto a single mode (the digit “7”), completely ignoring all other “modes” in the dataset (0, 1, 2, 3...).

## II. Solutions to Mode Collapse

### (1) WGAN

In the original GAN, the “adversarial” interaction is too intense, especially when discriminator $D$ becomes so good that generator $G$ cannot learn. Wasserstein GAN (WGAN) addresses this through several key changes.

1. Changing D's role and objective

In the original GAN, $D$ is a classifier outputting a probability between 0 and 1 (“this is fake” versus “this is real”). In WGAN, $D$ is redefined as a critic. It outputs a score rather than a probability, removing the original sigmoid function. The score can theoretically be any real number (such as $-10, 0, 50$). $D$ now aims to give real data $x$ the highest possible scores and forged data $G(z)$ the lowest possible scores. $G$'s objective changes accordingly: generate $G(z)$ that receives the highest possible score from $D$.

2. Gradient regularization

$D$ now seeks high scores for real data $x$ and low scores for forged data $G(z)$. However, relentlessly encouraging score separation makes $D$'s scoring curve “too steep,” causing exploding gradients. The original WGAN paper used weight clipping: inspect every weight and forcibly “pull it back” to $0.01$ if it exceeds a small range (such as $0.01$), or to $-0.01$ if it falls below $-0.01$. This causes vanishing gradients as gradients multiply across layers and limits capacity: to satisfy the constraint, $D$ “takes shortcuts,” pushing most weights to boundary values ($0.01$ or $-0.01$) and becoming a very simple function that can score only with words such as “good” or “bad,” losing the ability to learn complex, nuanced scoring curves.

We therefore choose the following mathematical expression as a penalty:

$$
\lambda \cdot (\lVert \nabla D(x) \rVert - 1)^2
$$

This forces $D$ to continually adjust its scoring curve during training so that its “slope” remains close to $1$ everywhere. Considering the actual computational cost, we usually evaluate the penalty at an intermediate point between real and forged data:

1. Sample: randomly draw a real image $x_{real}$ from the dataset.
2. Generate: have $G$ produce a forged image $x_{fake}$.
3. Interpolate: randomly choose an “intermediate point” $\hat{x}$ (read as x-hat) between these images.

   Mathematically, choose a random number $\epsilon$ between 0 and 1 and compute $\hat{x}=\epsilon\cdot x_{real}+(1-\epsilon)\cdot x_{fake}$. This is like randomly choosing a point on the line segment joining two points.

4. Check: compute the gradient penalty only at this “intermediate point” $\hat{x}$:

$$
(\lVert \nabla D(\hat{x}) \rVert - 1)^2
$$

Why does this “intermediate-point” strategy work? Imagine $D$'s “scoring landscape.” We force $D$ to maintain a smooth “hillside” of slope $1$ throughout the vast region between the “real-data territory” and the “forged-data territory.” Wherever $G$ is in the forged territory, it can clearly see a constant-slope uphill path toward the real territory. Under these conditions, the function's gradients inside both territories are also very likely to be close to $1$.

3. Training endpoint

Our earlier discussion of the GAN objective derived the following conclusion: for fixed $G$, the “optimal” $D^*$ maximizing $V(D,G)$ is:

$$
D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}
$$

When $G$ tries to minimize $\max_D V(D,G)$, it is actually minimizing $C(G)$:

$$
C(G)=2\cdot JSD(p_{data}\parallel p_g)-2\log 2
$$

Thus $G$'s objective (minimizing $C(G)$) is equivalent to minimizing JSD between $p_{data}$ and $p_g$.

When $p_{data}$ and $p_g$ do not overlap at all, JSD gets stuck at its constant maximum ($\log 2$), and its gradient becomes $0$. Proof:

KL divergence $D_{KL}(P\parallel Q)$ measures how much information is “lost” when switching from distribution $P$ to distribution $Q$:

$$
D_{KL}(P\parallel Q)=\sum_x P(x)\log\left(\frac{P(x)}{Q(x)}\right)
$$

JSD is simply a “smooth, symmetric” version of KL divergence. First define an “average distribution”:

$$
M=\frac{1}{2}(p_{data}+p_g)
$$

Then:

$$
JSD(p_{data}\parallel p_g)=\frac{1}{2}D_{KL}(p_{data}\parallel M)+\frac{1}{2}D_{KL}(p_g\parallel M)
$$

In the “nonoverlapping” case, first consider $D_{KL}(p_{data}\parallel M)$:

$$
D_{KL}(p_{data}\parallel M)=\sum_x p_{data}(x)\log\left(\frac{p_{data}(x)}{M(x)}\right)
$$

Substitute $M(x)=\frac{1}{2}(p_{data}(x)+p_g(x))$:

$$
D_{KL}(p_{data}\parallel M)=\sum_x p_{data}(x)\log\left(\frac{p_{data}(x)}{\frac{1}{2}(p_{data}(x)+p_g(x))}\right)
$$

Now consider only points where $p_{data}(x)>0$; elsewhere, $p_{data}(x)=0$ and the entire summand is 0.

At these points, since $p_{data}(x)>0$ and $p_g(x)=0$:

$$
\frac{p_{data}(x)}{\frac{1}{2}(p_{data}(x)+p_g(x))}=2
$$

Therefore:

$$
D_{KL}(p_{data}\parallel M)=\sum_x p_{data}(x)\log 2=\log 2
$$

Likewise, $D_{KL}(p_g\parallel M)=\log 2$, so $JSD(p_{data}\parallel p_g)=\log 2$ is constant. Consequently, $C(G)$ is constant, its gradient with respect to $G$ is 0, and it cannot drive changes in generator $G$'s parameters.

### (2) Noise Injection

“Noise injection” is not a single, universal solution to mode collapse; its effect depends on where the noise is added. Adding instance noise to discriminator inputs can smooth the real and generated distributions, alleviating training difficulties when their supports barely overlap. Injecting noise layer by layer into the generator in models such as StyleGAN primarily controls stochastic details such as hair and freckles, and does not justify claiming that it generally prevents mode collapse. Noise use should therefore specify the method and its objective; “adding noise to intermediate generator layers” must not be equated directly with a universal mechanism for preventing mode collapse.

## References

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets). NeurIPS 2014.
- Arjovsky, M., & Bottou, L. (2017). [Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/abs/1701.04862). ICLR 2017.
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). [Wasserstein GAN](https://arxiv.org/abs/1701.07875). ICML 2017.
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). [Improved Training of Wasserstein GANs](https://proceedings.neurips.cc/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html). NeurIPS 2017. (WGAN-GP)
- Karras, T., Laine, S., & Aila, T. (2019). [A Style-Based Generator Architecture for Generative Adversarial Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html). CVPR 2019.
