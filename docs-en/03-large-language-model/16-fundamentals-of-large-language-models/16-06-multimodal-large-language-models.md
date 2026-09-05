---
title: "16.6 Multimodal Large Language Models"
chapter_title: "Fundamentals of Large Language Models"
section_id: "16-06"
language: en
source_language: zh
source_docx: "第3部分 大语言模型/16.大语言模型的基本原理/16.6 多模态大语言模型.docx"
status: "auto-converted"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 16.6 Multimodal Large Language Models

## I. Text-Mediated Multimodal Perception

### 1. Basic Idea

This approach first uses perception models to convert images, audio, and other inputs into language (video is converted into images plus audio by sampling frames at intervals). For example, a pretrained vision encoder is trained independently rather than end to end within the entire LLM. Since the vision encoder and language model have different embedding spaces, the encoder output passes through an adapter before reaching the LLM as a stream of linguistically meaningful tokens that a text-only language model can consume directly.

The problem is that one-dimensional language cannot readily describe high-dimensional detail, so the adapter's conversion loses information. The overall model also cannot be optimized end to end.

### 2. Vision Encoder: CLIP as an Example

An input image is divided into a grid of fixed-size squares, usually 32×32 or 16×16 pixels. Each patch is flattened into a vector and converted to a visual token. Multi-head self-attention layers then produce a sequence of image-encoding vectors.

The vision encoder learns through “contrast”: matching (positive) pairs are pulled closer together, while mismatching (negative) pairs are pushed apart. For image–text pairs, an image and its corresponding text form a positive pair; unmatched image and text form a negative pair. Given a batch $(I_1,T_1),(I_2,T_2),\ldots,(I_n,T_n)$, the objective is to maximize similarity for matching pairs $(I_i,T_i)$ and minimize $s(I_i,T_j)$ for $j\ne i$, distinguishing paired images and texts from unpaired ones. The supervisory signal comes directly from the data: $(I_i,T_i)$ matches, while other random combinations $(I_i,T_j)$ are assumed likely not to match.

The loss is InfoNCE. Suppose there is one anchor sample $q$ (the query), one positive sample $k^+$ (the key matching $q$), and $N$ negative samples $k_1^-,\ldots,k_N^-$ (keys not matching $q$). We want the similarity between $q$ and $k^+$ to be much greater than that between $q$ and every $k_i^-$.

For one $q$, InfoNCE is:

$$
\mathcal{L}_q=-\log \frac{\exp(\mathrm{sim}(q,k^+)/\tau)}
{\exp(\mathrm{sim}(q,k^+)/\tau)+\sum_{i=1}^{N}\exp(\mathrm{sim}(q,k_i^-)/\tau)}
$$

$\mathrm{sim}(a,b)$: a similarity function, usually cosine similarity or dot product.

$\tau$: a temperature coefficient and hyperparameter, explained below.

The fraction:

$$
\frac{\exp(\mathrm{sim}(q,k^+)/\tau)}
{\exp(\mathrm{sim}(q,k^+)/\tau)+\sum_i\exp(\mathrm{sim}(q,k_i^-)/\tau)}
$$

uses Softmax regression to divide the matching score between $q$ and the positive sample by the total matching score between $q$ and all candidates (one positive and $N$ negatives). Given $q$ and these $N+1$ candidates, it is the predicted probability that $k^+$ is the correct match (positive sample). The aim is to make this probability approach 1.

$-\log$: cross-entropy loss. Suppose the true label is one-hot encoded as $[1,0,0,\ldots,0]$, meaning the first sample $k^+$ is positive and all others negative. Softmax outputs the model's predicted probability distribution, such as $[0.9,0.05,0.03,\ldots]$. Applying cross-entropy to this $N+1$-class classification problem gives the InfoNCE formula above.

InfoNCE converts contrastive learning (pulling together/pushing apart) into a classification problem of choosing one among $N+1$ candidates. Standard cross-entropy optimizes the model to identify the positive sample accurately among negatives, thereby “maximizing representational similarity between corresponding sequences and texts while minimizing similarity between negative pairs.” After training, an image of a dog has a vector representation close to the text vector for “a dog” and far from that for “a cat.”

Temperature $\tau$ controls Softmax's sharpness. At low temperature, similarity values are amplified, making the distribution very sharp: $\exp(10)$ is much larger than $\exp(5)$. The model is forced to push negatives away more strongly, including hard negatives that have some similarity to the anchor. This helps it learn finer-grained features. At high temperature, the distribution is smooth: $\exp(1)$ and $\exp(0.5)$ differ relatively little. Negative samples are penalized more mildly, and the resulting embeddings do not focus excessively on hard negatives.

## II. Native Multimodal Perception

### 1. Basic Idea

Images, audio, video, and other modalities are encoded into tokens alongside text tokens. Encoding both visual and textual information into token vectors participates in the LLM's end-to-end training and produces the same embedding space (for example, the word “apple” lies close to an image of an apple). Pretraining on mixed multimodal datasets enables direct understanding of deep cross-modal relationships, such as perceiving emotion directly from audio and turning it into a corresponding scene description, without passing through text as an intermediary.

### 2. Learnable Visual Embedding Layer

For an input image ($H\times W\times C$), first divide it into $N$ fixed-size patches, such as $14\times14$ pixels. The patch sequence $X_v=\{x_1,x_2,\ldots,x_N\}$ enters a learnable visual embedding layer and becomes a sequence of tokens (embedding vectors). These visual tokens then participate in mixed pretraining together with text tokens.

### 3. Mixed Pretraining

The model no longer receives only a text stream, but a stream mixing modalities. Suppose the input is “[image] is a cat.” The actual embedding sequence $S$ is:

$$
S=\mathrm{Concat}(Z_v^{(1)},\ldots,Z_v^{(N)},E_{text}^{(\mathrm{"is"})},E_{text}^{(\mathrm{"a"})},E_{text}^{(\mathrm{"cat"})})
$$

This long concatenated sequence $S$ enters the first Transformer layer directly.

At Transformer layer $l$, the input is $H^{(l)}$. The query ($Q$), key ($K$), and value ($V$) matrices are generated from the mixed sequence.

Suppose the model is generating a text token (the query) and needs to look back at image information in the context. Let $Q_{text}$ be the current text query, and $K_{img}$ and $V_{img}$ the key–value pairs of the image patches.

Attention no longer distinguishes modalities; it is computed jointly:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{Softmax}\left(\frac{Q\cdot [K_{text};K_{img}]^T}{\sqrt{d_k}}\right)\cdot [V_{text};V_{img}]
$$

Here, $[;]$ denotes concatenation along the sequence dimension.

- Relevance calculation ($QK^T$): when processing “cat,” $Q_{cat}$ computes dot products with $K_{img}$ for every patch. A patch containing “whiskers” or “pointed ears” produces a very large dot product.
- Information injection ($V$): Softmax weights are used to sum $V_{img}$ (visual image information), injecting it directly into the current text token's hidden state $H_{cat}^{(l)}$.

Thus, in layer 1, the text vector already absorbs visual vectors. In layer 2, the text vector containing mixed information interacts again with the original visual vectors for multi-step reasoning.

Training uses next-token prediction and still generates only text tokens at present (multimodality is limited to perception). Gradients pass through the entire architecture, including the visual embedding layer.

### 4. Multidimensional Positional Encoding

Images can use 2D positional encoding:

To address images' spatial relationships, Gemini extends RoPE to two dimensions.

For image-patch position $(x,y)$ and feature vector $v$, divide the vector into halves $v=[v_1,v_2]$. Define rotation as:

$$
\mathrm{RoPE}_{2D}(v,x,y)=\mathrm{Concat}(\mathrm{RoPE}_{1D}(v_1,x),\mathrm{RoPE}_{1D}(v_2,y))
$$

The attention score $Score_{ij}$ between two patches then depends not only on their content, but also on relative spatial distances $\Delta x$ and $\Delta y$. This enables understanding of geometric concepts such as “left” and “above.”

Video can instead use 3D positional encoding $(x,y,t)$.

### 5. Fine-Tuning

In supervised fine-tuning, high-quality instruction–answer pairs are often textual, so mixed multimodal training may be omitted. RL inputs are often mixed multimodal streams, matching the nature of environmental feedback for agents operating in the real world.

## III. Optimizing Image Processing

### (1) Semantically Aligned Complete Encoder (SAE) (LongCat-Next)

#### 1. Architecture

In SAE, input image $I$ does not follow a single deep processing channel throughout. Instead, a multilevel/dual-branch extraction strategy is used.

- Extracting explicitly semantically aligned features $F_{sem}$ (deep network): semantic features mainly answer “what” the image depicts. SAE uses deep Vision Transformer (ViT) blocks with multiple layers of global self-attention to abstract features of local pixel patches.

$$
F_{sem}=\mathrm{ViT}_{deep}(I)\in \mathbb{R}^{N\times D}
$$

- Extracting implicit low-level detail features $F_{res}$ (shallow/bypass network): low-level features answer “where” image details are and “how” they look. SAE branches off shallow or intermediate ViT layers through skip connections, or uses a lightweight auxiliary convolutional network. Shallow layers have not undergone excessive global pooling and retain high-frequency spatial structure, edge gradients, and original textures.

$$
F_{res}=\mathrm{ViT}_{shallow}(I)\in \mathbb{R}^{N\times D}
$$

Finally, dynamic gated fusion:

Suppose an MLP predicts the gate-weight matrix $G\in[0,1]^{N\times D}$:

$$
G=\sigma(W_g\cdot [F_{sem},F_{res}]+b_g)
$$

Here, $\sigma$ is Sigmoid activation and $[\cdot,\cdot]$ denotes channel concatenation. Final fusion uses Hadamard (element-wise) products:

$$
F_{out}=G\odot F_{sem}+(1-G)\odot F_{res}
$$

Interpretation: for a large area of uniformly colored sky, gating amplifies $F_{sem}$ (emphasizing “sky”) and suppresses redundant $F_{res}$. For dense text on a book page or hair, it strongly amplifies $F_{res}$ to ensure high-frequency textures reach the quantizer and remain unblurred in subsequent rendering.

#### 2. Loss Functions

(1) Independent alignment-training stage

Supervising $F_{sem}$: fine-grained dense semantic alignment

SAE replaces traditional global CLIP contrastive loss, which can lose local object information, with fine-grained region–text dense alignment. Let $T^{(i)}$ denote the text feature vector corresponding to image region $i$. Cross-entropy or contrastive learning maximizes their similarity:

$$
\mathcal{L}_{sem}=-\sum_{i=1}^{M}\log
\frac{\exp(\mathrm{sim}(F_{sem}^{(i)},T^{(i)})/\tau)}
{\sum_{j=1}^{K}\exp(\mathrm{sim}(F_{sem}^{(i)},T^{(j)})/\tau)}
$$

This supervision gives $F_{sem}$ strong linguistic affinity, making it easy for an LLM to interpret.

Supervising $F_{res}$: masked pixel-level reconstruction

To ensure $F_{res}$ truly remembers physical-world details implicitly for future image generation, SAE applies a reconstruction loss similar to that of a masked autoencoder (MAE) or lightweight diffusion model. The original high-frequency pixels $I_{target}$ must be reconstructed using $F_{res}$ alone:

$$
\mathcal{L}_{res}=\lVert D_{recon}(F_{res})-I_{target}\rVert_2^2
$$

The mean squared error (MSE) forces $F_{res}$ to retain implicit features such as lighting and materials that language cannot easily describe.

Separate supervision is insufficient. If $F_{res}$ secretly contains semantic information, or $F_{sem}$ redundantly records positions, “information channels become congested” during subsequent quantization.

To keep the two features mathematically pure, SAE introduces an orthogonal penalty. For each patch $i$, its semantic and detail vectors should be as perpendicular as possible in feature space (cosine similarity approaching 0):

$$
\mathcal{L}_{ortho}=\frac{1}{N}\sum_{i=1}^{N}
\frac{|\langle F_{sem}^{(i)},F_{res}^{(i)}\rangle|}
{\lVert F_{sem}^{(i)}\rVert_2\cdot \lVert F_{res}^{(i)}\rVert_2}
$$

Optimizing $\mathcal{L}_{total}=\mathcal{L}_{sem}+\lambda_1\mathcal{L}_{res}+\lambda_2\mathcal{L}_{ortho}$ physically isolates $F_{sem}$ and $F_{res}$ in two nonoverlapping subspaces.

(2) Native multimodal fusion stage

Only after SAE's “eyes” have fully developed is it connected to the MoE Transformer backbone for genuinely unified multimodal training.

- Global objective: the first stage's local loss $\mathcal{L}_{total}$ is discarded entirely. The whole system uses a single global objective: pure next-token prediction loss ($\mathcal{L}_{NTP}$).
- SAE's network state: most core SAE weights are frozen at this stage to protect the orthogonal feature space built in stage one from the language model's powerful gradients.
- Joint fine-tuning: usually only the projector between SAE and the LLM is trainable, or SAE is given an extremely small learning rate (possibly through a LoRA bypass) for minimal joint updates. Compute and gradients now focus on the LLM backbone, letting it learn how to “read” and “arrange” SAE's visual tokens.

### (2) Dynamic Resolution Processing

Suppose the native high-resolution input image is $I_{orig}\in\mathbb{R}^{H\times W\times 3}$, where $H$ and $W$ denote height and width. The standard input size accepted by the base Vision Transformer (ViT) encoder, or the size of one base block, is $S\times S$ (such as $336\times336$). To control computational complexity and memory use, the system sets an upper limit $N_{max}$ on the number of sub-blocks.

The first step is aspect-ratio-aware grid optimization. The model determines the original ratio $R=\frac{H}{W}$, then calculates the closest two-dimensional grid $(N_h,N_w)$, with $N_h$ rows and $N_w$ columns.

Among candidate integer pairs $\mathcal{G}$ satisfying $N_h\times N_w\le N_{max}$, it finds the combination closest to $R$. The optimization can minimize the logarithmic ratio difference:

$$
(N_h,N_w)=\arg\min_{(h,w)\in\mathcal{G}}\left|\log\left(\frac{h}{w}\right)-\log(R)\right|
$$

For a wide panorama with ratio $1:4$, this may produce a $1\times4$ grid; for a portrait phone screenshot, a $3\times1$ or $4\times2$ grid. The physical grid used for subsequent splitting therefore closely matches the original shape.

The second step is distortion-free proportional resizing and adaptive padding. Once $(N_h,N_w)$ is fixed, the target grid's total physical resolution is:

$$
H_{target}=N_h\times S
$$

$$
W_{target}=N_w\times S
$$

To fit the original image without stretching, calculate the proportional scale factor:

$$
\mathrm{Scale}=\min\left(\frac{H_{target}}{H},\frac{W_{target}}{W}\right)
$$

Resize the original proportionally to $(H',W')$:

$$
H'=\mathrm{round}(H\times\mathrm{Scale})
$$

$$
W'=\mathrm{round}(W\times\mathrm{Scale})
$$

Place $I_{resized}$ at the center or upper-left corner of the target grid. Since $H'\le H_{target}$ and $W'\le W_{target}$, remaining border regions are zero-padded using a background color such as the mean pixel value or pure black. This yields the standard image ready for splitting, $I_{padded}\in\mathbb{R}^{H_{target}\times W_{target}\times3}$.

## References

- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). ICML 2021.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR 2021.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). CVPR 2022.
- Meituan LongCat Team. (2026). [LongCat-Next: Lexicalizing Modalities as Discrete Tokens](https://arxiv.org/abs/2603.27538). arXiv:2603.27538.
