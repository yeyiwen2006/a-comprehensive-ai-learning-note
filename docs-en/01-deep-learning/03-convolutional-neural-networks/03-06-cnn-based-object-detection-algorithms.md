---
title: "3.6 CNN-Based Object Detection Algorithms"
chapter_title: "Convolutional Neural Networks"
section_id: "03-06"
language: en
source_language: zh
source_docx: "第1部分 深度学习/3.卷积神经网络/3.6 基于卷积神经网络的目标检测算法.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 3.6 CNN-Based Object Detection Algorithms

Convolutional neural networks (CNNs) and their variants have strong image-processing capabilities but cannot directly detect objects in images. The number of objects is uncertain: outputting a location and class for each object would make output dimensionality depend on the object count and therefore vary, whereas a CNN's output-layer dimensionality is fixed. This has motivated many CNN-based object-detection algorithms.

## I. Sliding-Window Method

Set windows of different sizes and aspect ratios and slide them with a chosen stride from the image's upper-left corner, extracting image patches. Resize each patch to a fixed size (such as 224*224), then feed it to a trained CNN classifier to determine whether the small window contains a "cat," "dog," or "background."

The drawback is that an image may produce thousands or tens of thousands of subwindows, each requiring a CNN pass, with massive redundant computation because neighboring windows overlap heavily. Moreover, the dimensionality issue is not resolved but avoided by running the model repeatedly: detection is forcibly decomposed into innumerable individual image-classification problems.

## II. Faster R-CNN: A Two-Stage Algorithm

Sliding windows repeatedly recompute overlapping pixels. Faster R-CNN passes the entire image through a CNN backbone (such as ResNet50) only once to obtain a feature map. All subsequent operations (RPN and classification) occur on this high-dimensional feature map, greatly saving computation.

1. Stage 1: Region Proposal Network (RPN)

The RPN addresses "where the boxes are" and is a fully convolutional network.

Input: the feature map extracted by the CNN, for example with size H*W*C.

Anchor mechanism: to handle objects with different sizes and aspect ratios, the RPN presets k anchors of different scales and ratios at each feature-map pixel (corresponding to a region of the original image). Typically k=9 (3 scales * 3 aspect ratios).

CNN convolution: apply a sliding 3*3 convolution to the feature map, followed by two 1*1 convolutional layers (equivalent to fully connected layers):

(1) Classification layer (cls layer): output 2k scores, the foreground/background probabilities for each anchor.

(2) Regression layer (reg layer): output 4k coordinate offsets, refining each anchor's x, y, w, and h.

Output: select high-scoring anchors and apply nonmaximum suppression (NMS) to obtain a series of regions of interest (RoIs).

2. Stage 2: RoI pooling and refinement

This addresses "uniform feature dimensions." RoIs from stage 1 vary in size, but subsequent fully connected layers require fixed-length inputs.

RoI pooling: divide the feature-map regions corresponding to differently sized RoIs into a fixed size (such as 7*7) through pooling, such as max pooling.

Final output: flatten each fixed-size feature block and feed it into fully connected layers for specific classification (N+1 classes, including background) and further bounding-box refinement.

## III. You Only Look Once (YOLO): A One-Stage Algorithm

YOLO treats object detection as regression. It completely abandons "propose first, classify second" and maps image pixels directly to bounding-box coordinates and class probabilities.

1. Input

YOLO divides the input image into S*S grid cells, resolving variable output dimensionality through this grid. A cell is responsible for detecting an object if the object's center falls within it. Whether an image contains 100 objects or none, YOLO's output size is always fixed.

Each cell predicts the class of an object centered within it. During training, if an object's center falls in a cell, that cell's desired output is labeled with the object's x, y, w, h, class, and related information; this produces the same "division of responsibility" at test time. For objects assigned to each cell, B possible bounding boxes are predicted (typically 2 in YOLO v1).

2. Output

The data form an S*S*(5*B+Class)-dimensional tensor with three parts:

The first contains each box's center coordinates x,y and dimensions w,h, totaling S*S*4*B values.

The second contains each box's confidence. During training, the desired output is Pr*IOU, where Pr is 1 if the box contains an object and 0 otherwise. During training, IOU=(area of predicted box ∩ actual box)/(area of predicted box U actual box). At test time, the output is essentially a fit to this training target.

The third contains each cell's predicted probability of belonging to each class. With Class classes, there are S*S*Class values.

3. Loss function

YOLO's loss function is key to successful training. It is a weighted sum of localization, confidence, and classification errors.

(1) Localization loss: use mean squared loss for x,y and for sqrt(h),sqrt(w), preventing large-object errors from dominating. The same pixel deviation (such as predicting a box 5 pixels too wide) is minor for a large box but a substantial error for a small one.

(2) Confidence loss: use mean squared loss, multiplying cells containing object centers by a larger coefficient (such as 5) and cells without object centers by a smaller one (such as 0.5), because object-detection confidence matters more than background.

(3) Classification loss: v1 uses mean squared loss; v2 and later use cross-entropy loss.

## References

- Ren, S., He, K., Girshick, R., & Sun, J. (2015). [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html). NeurIPS 2015.
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640). CVPR 2016.
- Redmon, J., & Farhadi, A. (2017). [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242). CVPR 2017.
- Redmon, J., & Farhadi, A. (2018). [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767). arXiv:1804.02767.
