---
title: "26.3 Diffusion U-Net"
chapter_title: "Diffusion Models"
section_id: "26-03"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/26.扩散模型/26.3 Diffusion U-Net.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 26.3 Diffusion U-Net

U-Net uses a convolutional neural network. Its first half resembles an encoder: pooling or convolutions with a stride greater than 1 progressively reduce the spatial dimensions while increasing the number of channels. Its second half resembles a decoder: the spatial dimensions progressively increase while the number of channels decreases. There are two main ways to implement this:

**1. Transposed convolution**

This is the core upsampling technique used in the U-Net decoder. It is not simply image interpolation, but a reverse spatial spreading process with learnable parameters.

Standard convolution can be represented as the matrix multiplication $\mathbf{Y}=\mathbf{C}\mathbf{X}$, where $\mathbf{C}$ is a sparse matrix derived from the convolution kernel. This process compresses a higher-dimensional $\mathbf{X}$ into a lower-dimensional $\mathbf{Y}$.

Transposed convolution instead maps the output feature map back into a larger space by multiplying it by the transpose $\mathbf{C}^T$ of $\mathbf{C}$ (and learning the weights through backpropagation):

$$
\mathbf{X}'=\mathbf{C}^T\mathbf{Y}
$$

Workflow:

Consider a kernel size $K=3$, stride $S=2$, and padding $P=1$:

1. **Internal zero-padding expansion**: insert $S-1$ zeros between adjacent pixels in the input feature map. This expands the spatial dimensions of the low-resolution input.
2. **Boundary padding**: add extra zero padding around the expanded feature map (with the amount determined by $K$ and $P$ to match the required output dimensions exactly).
3. **Standard convolution mapping**: apply a standard $K\times K$ convolution kernel to the large feature map produced by these two padding operations (the convolution stride is now fixed at 1).
4. **Feature output**: as the kernel slides over the feature map, it multiplies the original isolated pixel values by its weights and “spreads” them into the surrounding zero-filled areas, ultimately producing a smooth, high-resolution feature map.

**2. Bilinear interpolation**

This is a parameter-free, purely mathematical interpolation method. It generates each new pixel by calculating a weighted average of the four known pixels surrounding its corresponding position in the original image. Its computational cost is very low, but unlike transposed convolution, it cannot adaptively learn upsampling rules from the characteristics of the dataset.

However, U-Net has limited scalability and is incompatible with current Transformer-based large-model architectures.

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS.
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). CVPR.
