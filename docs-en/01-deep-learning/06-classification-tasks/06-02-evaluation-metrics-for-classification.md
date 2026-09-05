---
title: "6.2 Evaluation Metrics for Classification"
chapter_title: "Classification Tasks"
section_id: "06-02"
language: en
source_language: zh
source_docx: "第1部分 深度学习/6.分类任务/6.2 分类任务的评估标准.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 6.2 Evaluation Metrics for Classification

## I. Why Do We Need Multiple Evaluation Metrics?

During training, cross-entropy loss and Softmax regression guide gradient descent and parameter adjustment. After training, parameters are fixed, and we need a "human-readable" report card to judge practical performance. A cross-entropy value such as 0.43 is not intuitive enough. We want to know: "What kinds of mistakes will this model make in my task, and how much can I trust it?"

Accuracy alone is insufficient. Suppose a classifier detects NAD+-binding proteins, which constitute 0.05% of the samples. A model that takes a shortcut and predicts every protein as "not NAD+-binding" achieves 99.95% accuracy (0.05% incorrect and 99.95% correct), yet is useless because it finds none of the proteins we want.

## II. Core Concepts

### 1. Precision

Meaning: among all results predicted "yes" (positive), the proportion that truly are "yes."

Formula:

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

True positive (TP): truly "yes," predicted "yes."

False positive (FP): actually "no," but incorrectly predicted "yes."

Goal: reduce false positives and avoid classifying false samples as true—"better fewer than incorrect." In spam detection, high precision means that a large proportion of messages flagged as spam genuinely are spam, with few legitimate messages wrongly flagged.

### 2. Recall

Meaning: among all truly positive samples, the proportion successfully predicted "yes."

Formula:

$$
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

False negative (FN): truly "yes," but incorrectly predicted "no"—a missed case.

Goal: reduce missed cases, "preferring a thousand false alarms to missing one target." In infectious-disease screening, high recall means the model successfully identifies the vast majority of actual patients, with few missed diagnoses, finding everyone who should be found.

### 3. The Precision–Recall Trade-Off

These metrics usually conflict. A model outputs a probability score (such as 0%–100%), not simply "yes" or "no." A decision threshold (such as 50%) determines when the score counts as "yes." A very high threshold, such as 99%, makes the model "cautious": precision is high, but FN is high and recall low. A low threshold, such as 10%, makes it "lenient," predicting "yes" whenever there is even a small possibility. Recall is high, but FP is also high (many negatives are misclassified), giving low precision.

### 4. The Precision–Recall Curve

Sweep the decision threshold from high to low, record each corresponding (recall, precision) point, and connect them.

X axis: recall.

Y axis: precision.

At a threshold of 100%, the model predicts nothing as "yes," giving recall 0. Gradually lowering the threshold (99%, 98%, and so forth) produces more positive predictions, so recall increases monotonically or remains unchanged. Precision usually fluctuates downward. Plotting and connecting all (recall, precision) points gives the P–R curve.

The P–R curve is a powerful evaluation tool, especially for imbalanced data (such as 99% negative and 1% positive):

Ideal model: a perfect curve reaches the top-right corner (1, 1), maintaining 100% precision (no incorrect predictions) at 100% recall (all positives found).

Excellent model: the curve stays as close as possible to the top-right corner.

Poor model: the curve lies close to the bottom-left corner.

## III. Common Metrics Based on These Concepts

### 1. Area Under the Precision–Recall Curve (AUPR)

For a correct pairing, compute the model's precision and recall for that pairing and the area beneath its P–R curve. A larger area, closer to 1.0, indicates stronger overall performance across thresholds.

When many samples can be paired in multiple ways (such as one-to-many protein–function pairings), each pairing has an AUPR value, and these values are often averaged over all pairing types.

Mean AUPR (sometimes simply called AUPR) measures global average performance across pairing decisions: it is "pair-centered." It does not focus on individual paired objects, but evaluates millions of "Does (A, B) match?" decisions.

For protein–function pairing, for example, it "flattens" all possible function predictions for all proteins and computes one overall precision–recall area. It asks: "How well does the model perform across all its function-assignment decisions?" Under this calculation, common functions—those occurring in many proteins—have greater influence on the total AUPR.

### 2. Maximum F-Score (F-max)

$F_{\max}$ is "centered on the objects being paired."

Consider protein–function pairing. The output for each pair is a probability from 0 to 1, requiring a threshold, such as 0.5, to decide whether to assign that function.

(1) First, evaluate each protein at a given threshold $t$ using the $F$-measure ($F$-score), the harmonic mean of precision and recall.

Why the harmonic mean? It penalizes extreme weaknesses more harshly. It is high only when both precision and recall are high; if either is poor, it gives a much lower value than the arithmetic mean. In classification, particularly imbalanced tasks such as protein-function annotation, severely uneven performance is undesirable. Suppose model A at threshold $t$ has precision $P=1.0$ (every prediction correct) but recall $R=0.01$ (missing $99\%$ of targets). It is almost useless in practice because it finds virtually nothing.

(2) Next, average the $F$-scores over all proteins, producing a function of $t$. Try every possible threshold $t$ and find the $t$ maximizing the mean $F$-score. Report this "best-case" mean $F$-score as $F_{\max}$.

Here, $F_{\max}$ asks: "For an ordinary protein, how accurately can the model predict its function list at its best-performing threshold?" It reveals average performance for individual proteins, regardless of whether their functions are rare.

### 3. Combining the Two

A good model must excel on both dimensions. For example, it could have high $F_{\max}$ (correctly guessing several functions per protein on average) but low AUPR (making many small errors across common functions). Such a model is not good enough.

## References

- Davis, J., & Goadrich, M. (2006). [The Relationship Between Precision-Recall and ROC Curves](https://doi.org/10.1145/1143844.1143874). ICML 2006.
- Radivojac, P., Clark, W. T., Oron, T. R., et al. (2013). [A large-scale evaluation of computational protein function prediction](https://www.nature.com/articles/nmeth.2340). Nature Methods.
