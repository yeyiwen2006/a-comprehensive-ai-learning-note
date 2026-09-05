---
title: "3.1 Convolutional Neural Networks (CNNs)"
chapter_title: "Convolutional Neural Networks"
section_id: "03-01"
language: en
source_language: zh
source_docx: "第1部分 深度学习/3.卷积神经网络/3.1 卷积神经网络（CNN）.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 3.1 Convolutional Neural Networks (CNNs)

## I. Core Idea: Hierarchical Feature Extraction Inspired by Human Vision

When recognizing a cat, for example, you first notice local details: edges (whiskers and ear outlines), spots, and patches of color. These details then combine into larger structures: eyes, a nose, and paws. Finally, these structures combine into an overall concept: a cat.

## II. Core Components

1. Convolutional layer: sliding scans by feature detectors

(1) Intuitive explanation

Imagine slowly sliding a magnifying glass (a "filter" or "convolution kernel") across an image from left to right and top to bottom. This magnifying glass has a special ability: it "lights up" particular patterns it sees (such as vertical lines, horizontal lines, or particular colors). At every position, it computes a value indicating how well that location matches the pattern it detects. Once all positions have been processed, the result is a new "feature map," highlighting every region of the original image that matches the pattern. A CNN typically has many different magnifying glasses (filters), each detecting a different pattern (edges, textures, shapes, and so forth), producing multiple feature maps.

Essentially, each value in a convolution kernel is a shared weight between certain inputs and outputs. An output unit is connected only to a local input region (its receptive field), rather than being fully connected. A filter uses the same weights at different input positions. Thus, wherever a feature appears in the image, it is detected by the same detector. This underlies CNNs' translation invariance (position invariance) and is central to reducing parameters.

(2) Input

Each input is a four-dimensional tensor. Under PyTorch conventions, the first dimension is batch_size; the second is input_channels (the number of input channels—for example, the first convolutional layer for an RGB image has 3 input channels representing the three colors, while subsequent layers have as many input channels as the number of features produced by the preceding layer; kernel depth must match the image's channel count); the third and fourth dimensions are image height and width, input_h and input_w.

(3) Convolution kernel

Also called a filter, this is a small, learnable weight matrix. The filter matrix has three dimensions: height kernel_h, width kernel_w, and channel count input_channels. A convolutional layer usually has multiple filters; their number is the output-channel count output_channels. Each filter detects a particular local spatial pattern and corresponds to one output channel.

(4) Convolution operation

The output matrix is a four-dimensional tensor: the first dimension is batch_size, the second is output_channels (the number of kernels), and the third and fourth are output_h and output_w.

For each sample in the batch, the filter slides over the input (with stride paddle). At each position, it computes a dot product (Σ(each kernel element * the corresponding input-sample element), followed by summation across channels), possibly adds a bias, and produces the output for that sample, channel, and position.

(5) Low-level implementation of convolution

Because parallel computation is far more efficient than for loops, we want to express convolution as matrix multiplication A*B, where A represents the input and B the kernels. Since the input is four-dimensional and the kernels are three-dimensional, some dimensions are merged and flattened. Observing the connection between matrix multiplication and dot products, we want the number of columns of A = the number of rows of B = the dimension of the vectors in the dot product. This dimension is kernel_h*kernel_w*input_channels, so A and B are constructed as follows:

Rows of A = batch_size (viewed as concatenating batch_size matrices) * output_h*output_w (the number of positions at which convolution is performed, hence the number of convolution operations).

Columns of A = rows of B = input_channels*kernel_h*kernel_w (convolution sums over rows, columns, and channels).

Columns of B = output_channels (viewed as concatenating matrices for output_channels distinct filters, each detecting a different feature).

B is therefore the weight matrix, but A is not the original input matrix: it requires transforming that input.

For sample b, with the upper-left corner of the kernel at [i,j], crop input_data[b,i:i+kernel_h,j:j+kernel_w,:] from the input matrix, flatten it, and concatenate all such matrices. For efficiency, we generally use the unfold function. For each position reachable by a kernel center (a row), it collects all kernel_h*kernel_w elements in the kernel-covered region of each channel (columns). To extract image patches with unfold, first slide a kernel_h*1 "kernel" along dimension 2 (height) to "collect the elements at the kernel center." There are output_h*input_w reachable kernel-center positions, giving shape [batch_size, input_channels, output_h, input_w, kernel_h]. Then operate along dimension 3 (width) of the result, giving shape [batch_size, input_channels, output_h, output_w, kernel_h, kernel_w].

A bias is sometimes added; each kernel (each output channel) has only one bias.

2. Activation function: introducing nonlinearity

The values computed by a convolutional layer (feature-match scores) are generally linear. Real-world data and decisions, however, are often highly nonlinear. An activation function resembles a "switch" or "threshold processor," deciding whether a feature is strong enough to pass to the next layer. It adds nonlinear processing capability, allowing the network to learn complex mappings.

3. Pooling layer: downsampling and compressing spatial information

A pooling layer is an "information concentrator." Within a small region (such as a 2x2 square), it selects a representative value (such as the maximum or average). Given an input feature map, pooling-window size P_h*P_w, and stride S (usually equal to the window size, so windows do not overlap), the window slides across the feature map in steps of S. A representative value is computed within each window to form the output feature map. This operation is independent for each channel and has no learnable parameters.

(1) Types of pooling

Max pooling: take the maximum of all elements in the window.

Average pooling: take the average of all elements in the window.

(2) Benefits of pooling

Dimensionality reduction (keeping the big picture): pooling divides a feature map into small regions (such as 2x2 pixels) and selects one "representative" from each (the maximum or average). This resembles reducing a high-resolution photograph to a thumbnail: details are lost, but the cat's outline and main features remain. It greatly reduces the data processed by subsequent layers, lowering computation and memory usage and preventing an overly complex network.

Improved robustness (reduced sensitivity to position): if a cat in an image moves slightly (by a few pixels), feature positions in the convolutional output move too. Pooling, however, cares only about the "loudest note" (maximum) or "overall tone" (average) in a small region. As long as the strongest feature there is still a cat's ear, the pooled output remains stable. This makes the network less sensitive to small translations, rotations, and scaling changes (translation invariance), improving generalization.

Preventing overfitting (blurring details): discarding precise locations and nonessential details reduces sensitivity to particular noise or irrelevant training-data details. The model focuses on whether a feature is present rather than its exact pixel location.

Expanding the receptive field (seeing more): as the network deepens, subsequent convolutional layers act on pooled feature maps, effectively "seeing" larger regions of the original image (larger receptive fields). This helps capture higher-level semantics, such as an entire cat's head rather than a single whisker.

(3) Modern improvements

Learnable or smooth pooling rules: research uses learnable parameters or smooth weighting so that pooling is no longer fixed to max or mean. SoftPool, for example, computes a weighted sum of local features using exponential weights based on activation values.

Combining attention: use attention mechanisms to dynamically adjust the importance weights of pooling regions, selecting or integrating information more intelligently.

Dilated convolution: enlarge a kernel's receptive field (by inserting gaps between elements) to avoid downsampling with pooling layers. This is common in tasks requiring high-resolution outputs, such as semantic segmentation.

4. Fully connected layer: integrated decisions

In the final layers of a CNN, feature maps obtained after repeated convolution, activation, and pooling (possibly flattened into a one-dimensional vector) enter one or more fully connected layers. These resemble a conventional neural network (a multilayer perceptron, MLP): each neuron connects to every neuron in the preceding layer. Their role is to combine all the high-level abstract features extracted earlier and make a final decision, such as the probability that the image is a "cat" or "dog."

5. Other important concepts

(1) Padding: add a border of pixels (usually zeros) around the image so that edge information can also occupy the center of a convolution region. This gives edge and central information "equal treatment" during convolution and controls the output feature-map size so that it remains unchanged.

(2) Stride: the number of pixels a filter moves horizontally and vertically at each step. Larger strides mean faster sliding, smaller output feature maps, and faster growth of the receptive field.

(3) Batch normalization: during training, normalize the input to each layer (usually after a convolutional or fully connected layer and before activation) by subtracting the mean and dividing by the standard deviation, then introduce learnable scaling and shifting parameters. This accelerates training, alleviates vanishing/exploding gradients, permits higher learning rates, and provides some regularization.

For mini-batch inputs $B=\{x_1,\ldots,x_m\}$, compute the mini-batch mean and variance:

$$
\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i,\qquad
\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

Normalization, scaling, and shifting are:

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad
y_i=\gamma\hat{x}_i+\beta
$$

Here, $\epsilon$ is a small constant preventing division by zero; $\gamma$ and $\beta$ are learnable parameters that restore the network's representational capacity.

(4) Dropout: during training, randomly make some neurons temporarily "inactive" (set their outputs to zero). This prevents excessive dependence on particular neurons, improves generalization, and reduces overfitting. Each neuron's output is zeroed with probability p during training. At test time, all neurons are active, but their outputs are multiplied by 1-p (or multiplied by 1/(1-p) during training and left unchanged at test time) to preserve the expected output.

6. Multiple channels

Input data and feature maps are not actually single two-dimensional matrices ($H \times W$), but three-dimensional tensors: $H(\mathrm{Height}) \times W(\mathrm{Width}) \times C(\mathrm{Channels})$.

A standard RGB color image is an H*W*3 input matrix with three channels: red (R), green (G), and blue (B). Each channel is a two-dimensional grid (matrix), with each value (pixel) indicating the intensity of that color (R/G/B) at that location. A grayscale image has only one channel (brightness).

For feature maps produced after CNN convolutional layers, the channel count represents different types of features. For example, one output channel of the first layer might be primarily sensitive to "vertical edges," while another detects "45-degree edges" or "red regions." Deeper in the network, channels represent increasingly abstract features (such as patterns associated with "wheels" or "cat ears"). A convolutional layer's output feature map has dimensions H*W*C_out. Its output-channel count C_out equals the number of kernels used, hence the number of distinct feature types.

Convolution with a kernel at position (i, j) proceeds as follows:

(1) Extract the input patch: from input X, take a three-dimensional block X[i:i+K_h, j:j+K_w, :] matching the kernel size. This block has dimensions (K_h, K_w, C_in). Input-channel count and kernel depth must match (both are C_in), enabling multichannel convolution to integrate information across channels. The kernel "scans" across the channel dimension and combines information from every input channel.

(2) Channelwise dot products and summation: multiply the kernel and input patch elementwise, producing a three-dimensional result of the same size (K_h, K_w, C_in). For RGB inputs, a kernel detecting "edges of red objects" might assign positive weights to R and negative or zero weights to G and B. In deeper feature maps, a "wheel" detector may combine information from lower-level feature channels such as "circular shapes," "black regions," and "textures."

(3) Global summation and bias: sum all elements of this three-dimensional result and add a scalar bias b_k (unique to each kernel). This aggregates the correlations computed at every spatial location and input channel into one scalar. It represents the overall response strength at position (i, j) for the kernel's particular feature, such as an edge, texture, or color combination.

(4) Obtain the output: this (Sum + b_k) is the value of output feature map O at position (i, j) for kernel k (output channel k): O[i, j, k] = b_k + sum_{di=0}^{K_h-1} sum_{dj=0}^{K_w-1} sum_{c=0}^{C_in-1} ( X[i+di, j+dj, c] * K[di, dj, c, k] ).

## III. Typical Architecture Pattern

Classic CNN architectures generally follow this pattern:

`Input image -> [[Convolutional layer -> Activation] * N -> Pooling layer] * M -> Flatten -> [Fully connected layer -> Activation] * K -> Output layer`

As depth increases, spatial resolution gradually decreases (through pooling and large-stride convolutions), while the number of channels grows (using more distinct filters). Extracted features evolve from low-level features (edges, corners, and textures) to high-level ones (object parts, complete objects, and scenes). The final fully connected layers combine these high-level features to perform the task, such as classification or regression.

A classic convolutional neural network example: LeNet

LeNet-5 (the classic version) has 7 layers, takes $32\times 32$ grayscale images as input, and outputs a probability distribution over digits 0–9:

| Layer | Operation | Input size | Output size | Main role |
|---|---|---|---|---|
| Input | Original image | $32\times 32\times 1$ | $32\times 32\times 1$ | Receive a grayscale image |
| C1 convolution | Six $5\times 5$ kernels | $32\times 32\times 1$ | $28\times 28\times 6$ | Extract edge/texture features |
| S2 pooling | $2\times 2$ average pooling | $28\times 28\times 6$ | $14\times 14\times 6$ | Reduce dimensions and retain key features |
| C3 convolution | Sixteen $5\times 5$ kernels | $14\times 14\times 6$ | $10\times 10\times 16$ | Extract complex features, such as local digit structures |
| S4 pooling | $2\times 2$ average pooling | $10\times 10\times 16$ | $5\times 5\times 16$ | Further reduce dimensions |
| C5 fully connected | Flatten + fully connected | $5\times 5\times 16\rightarrow 400$ | 120 | Combine high-level features |
| F6 fully connected | Fully connected | 120 | 84 | Nonlinear feature mapping |
| Output | Fully connected + Softmax | 84 | 10 | Produce class probabilities |

## IV. Advantages of CNNs

1. Parameter sharing: one filter slides across the entire image with the same weights, greatly reducing learnable parameters compared with a fully connected network. Training becomes more efficient, models become smaller, relatively less training data are required, and translation invariance is provided.

2. Local connectivity: output units connect only to local input regions, matching the local correlations of image data (a pixel's semantics are usually determined by neighboring pixels) and reducing parameters.

3. Hierarchical feature extraction: learn increasingly complex features directly from raw pixels (edges -> textures -> parts -> objects), without manually designed features.

4. Translation invariance: parameter sharing and pooling give the network a degree of insensitivity to an object's changing position within the image. Where a feature appears is unimportant; what matters is that it appears.

5. Spatial hierarchy: convolution and pooling naturally establish a spatial hierarchy in the network structure.

## V. Trends in CNN Architectures

1. Fewer pooling layers, replacing their downsampling with convolutions whose stride exceeds 1.

2. Extensive use of batch normalization.

3. Deeper networks (ResNet, DenseNet, and others use skip connections to address the difficulty of training deep networks).

4. Smaller filters (such as stacked 3x3 convolutions instead of large kernels).

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). Proceedings of the IEEE.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). NeurIPS 2012.
- Stergiou, A., Poppe, R., & Kalliatakis, G. (2021). [Refining Activation Downsampling with SoftPool](https://arxiv.org/abs/2101.00440). ICCV 2021.
