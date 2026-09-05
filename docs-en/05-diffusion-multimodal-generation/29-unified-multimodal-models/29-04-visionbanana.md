---
title: "29.4 VisionBanana: Unifying Understanding Tasks through Visual Generation"
chapter_title: "Unified Multimodal Understanding and Generation Models"
section_id: "29-04"
language: en
source_language: zh
source_docx: "第5部分 扩散模型与多模态生成/29.统一多模态理解-生成模型/29.4 VisionBanana：用视觉生成统一理解任务.docx"
status: "manually reconstructed from Word-visible content"
ocr: "not used; Word-visible images manually classified and reconstructed"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 29.4 VisionBanana: Unifying Understanding Tasks through Visual Generation

## I. Background

Image generation and image understanding remain difficult to unify within one model, and their gradients are often found to conflict in practice. Yet this runs counter to our intuition: during training, image-generation models must learn objects, scenes, geometry, semantics, materials, spatial relationships, and so on, which already encompass image understanding.

For visual-understanding tasks such as segmentation, depth, and normals, earlier work observed that generative models could occasionally draw something resembling a “depth map” or “segmentation map.” However, these output formats are unstable, cannot be reliably decoded into standard visual-task results, and are difficult to evaluate fairly against specialized models.

## II. Core Idea

The core contribution of “Image Generators are Generalist Vision Learners” is to reformulate all visual-understanding tasks as “generating an RGB image,” then have an image generator produce decodable visual results according to instructions. The authors apply lightweight instruction fine-tuning to the image-generation model Nano Banana Pro to obtain Vision Banana. Given an original image and a natural-language instruction, it outputs an RGB image, which is then converted back into standard visual-task results, such as masks, depth values, or normal directions. For example, semantic segmentation no longer requires the model to output class logits. Instead, it generates a colored segmentation map: “Please generate a semantic-segmentation visualization of this image: use red for roads, blue for the sky, yellow for pedestrians, and black for the background.” After the model generates the RGB image, the system decodes each pixel into a class according to its color. In this way, segmentation becomes an image-generation task.

This closely resembles the large-language-model paradigm, in which LLMs use text generation to handle translation, question answering, reasoning, coding, and other tasks uniformly. This paper attempts to show that image generation can likewise serve as a unified interface for visual-understanding tasks. Segmentation, depth, and normal maps can all be encoded as RGB images, allowing one generative model to handle multiple tasks through the same output format. Vision Banana does not design a new head, network, or loss for each task. It shares one generative model and changes only the prompt and output color-coding rules. The paper reports that Vision Banana reaches or approaches specialized-model performance on multiple tasks.

## III. Task Categories

### (1) 2D Semantic Understanding

Semantic segmentation: Assign a semantic class to each pixel, such as road, person, sky, or car.
Instance segmentation: Separate different object instances of the same class; for example, five dogs should have five distinct masks.
Referring-expression segmentation: Identify a target region from a natural-language description, such as “the man wearing a pink T-shirt,” “the cat stretching,” or “the chef's names in Chinese and English on the menu.”

### (2) 3D Geometric Understanding

Monocular metric depth estimation: Estimate the actual physical distance from each pixel to the camera, in meters, from a single RGB image.
Surface normal estimation: Estimate the orientation of the surface at each pixel, that is, the local 3D geometric direction.

## IV. Training

Vision Banana is obtained by lightweight instruction fine-tuning of Nano Banana Pro. Its training data mixes two components:

Original image-generation training data: Preserve the model's original image-generation capability.

A small amount of visual-task data: Teach the model to output visual-task results as RGB images in specified formats.

The paper particularly emphasizes the very low proportion of visual-task data. The purpose is not to train a segmentation or depth model from scratch, but to teach the generator to “understand the task format” and turn its existing internal visual representations into measurable outputs. The authors aim to show that visual-understanding capability comes primarily from generative pretraining rather than large amounts of task-specific fine-tuning.

## V. Encoding Rules

Segmentation is the simplest case. Suppose the model-generated image is:

$$
G(x, y) \in [0, 255]^3
$$

Here, $G(x, y)$ denotes the RGB color of pixel $(x, y)$. If the prompt specifies that class $c$ corresponds to color $q_c$, decoding can use nearest-color matching:

$$
\hat{c}(x, y) = \arg\min_c \left\|G(x, y) - q_c\right\|_2
$$

In other words, for each pixel, find the class color closest to its generated color and assign the pixel to that class.

Instance segmentation is slightly different because the number of instances is unknown beforehand, so the color of each instance cannot be specified in the prompt in advance. The paper uses per-class inference: at each pass, the model segments only one class, for example, “represent each garlic instance with a different color.” The model assigns colors to different instances itself, and post-processing clusters by color to separate differently colored regions into different instances.

Depth estimation is more complex because depth is real-valued:

$$
d \in [0, \infty)
$$

Whereas RGB has a finite range:

$$
RGB \in [0, 1]^3
$$

The authors therefore design an invertible mapping that encodes actual depth $d$ as an RGB color. The paper first uses Barron's power transform to compress depth into $[0, 1)$:

$$
f(d, \lambda, c) = 1 - \left(1 - \frac{d}{\lambda c}\right)^{\lambda + 1}
$$

Here, $d$ is actual metric depth in meters; $\lambda$ is a shape parameter, set to $-3$ in the experiments; $c$ is a scale parameter, set to $10/3$; and $f(d,\lambda,c)$ is normalized depth, used to map onto an RGB color path.

The reason is that depth errors for nearby objects are usually more important, so the mapping allocates higher resolution to short distances while compressing long distances. The paper then maps this normalized depth to colors along the edges of the RGB cube. During inference, the model generates a colored depth map, and an inverse function decodes the colors back into metric depth.

Surface normals are more straightforward. A normal is a unit vector:

$$
\mathbf{n} = (x, y, z), \quad x, y, z \in [-1, 1]
$$

RGB channels lie in $[0, 1]$, so a linear transformation can be used:

$$
RGB = \frac{\mathbf{n} + 1}{2}
$$

That is:

$$
R = \frac{x + 1}{2}, \quad G = \frac{y + 1}{2}, \quad B = \frac{z + 1}{2}
$$

After the image is generated, invert the transformation:

$$
\mathbf{n} = 2RGB - 1
$$

The RGB image can thus represent the surface orientation at every pixel.

## References

- Gabeur, V., Long, S., Peng, S., Voigtlaender, P., Sun, S., Bao, Y., Truong, K., Wang, Z., Zhou, W., Barron, J. T., Genova, K., Kannen, N., Ben, S., Li, Y., Guo, M., Yogin, S., Gu, Y., Chen, H., Wang, O., Xie, S., Zhou, H., He, K., Funkhouser, T., Alayrac, J.-B., & Soricut, R. (2026). [Image Generators are Generalist Vision Learners](https://arxiv.org/abs/2604.20329). arXiv:2604.20329.
