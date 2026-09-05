---
title: "9.1 Graph Neural Networks"
chapter_title: "Graph Neural Networks"
section_id: "09-01"
language: en
source_language: zh
source_docx: "第1部分 深度学习/9.图神经网络/9.1 图神经网络.docx"
status: "translated"
ocr: "no pending image placeholders in public Markdown"
license: "CC BY-NC-SA 4.0"
local_only: false
---

# 9.1 Graph Neural Networks

## I. Basic Principles of Graph Neural Networks (GNNs)

### (1) Forward Algorithm

The core of a GNN is message passing, analogous to “neighbors exchanging information to update their understanding.” Its mathematical workflow has three steps:

1. Message generation: for an edge e(u,v) connecting nodes u and v, a “message” m_uv can be generated. The message-generation function usually considers node u's features h_u, node v's features h_v, and edge e(u,v)'s features f_e(u,v) together. For example, m_uv = Message(h_u, f_e(u,v), h_v).

2. Message aggregation: each node v receives messages m_uv from all neighboring nodes u and aggregates them into M_v=Aggregate({m_uv|u∈N(v)}). Aggregate is usually a sum, mean, maximum, or similar function.

3. Node update: node v's features h_v are updated using its previous features h_v and aggregated message M_v. The update function can be written h_v(t+1) = Update(h_v(t), M_v(t)).

### (2) Iterative Process

The steps above repeat K times (K GNN layers), allowing nodes to capture K-hop neighborhood information:

Layer 1: learns direct neighbors' features.

Layer 2: neighbors learned their own neighbors' features in layer 1, so this layer learns about “neighbors of neighbors.”

Note: too many layers can cause oversmoothing (all node representations become similar).

### (3) Backpropagation

Message-generation, aggregation, and update functions are usually implemented by neural networks (such as multilayer perceptrons, MLPs) with learnable parameters (weight matrices and biases). During backpropagation, gradients for these parameters are computed according to the loss function's effect on model outputs, and node features are updated.

Some advanced models also allow edge features to be updated dynamically during message passing. For example, when a multiscale cross-attention Transformer learns molecular features, both node and edge features of the molecular graph are converted into input embeddings that Transformer layers can process, and are processed through multiple Transformer layers. Another example is intramolecular message passing, whose initial state includes node features and edge features related to bond stretching and bond angles; both node and edge features are updated at every message-passing layer. Edge updates usually resemble node updates: a function may generate new edge features from connected nodes' features and previous edge features, or a dedicated edge-attention mechanism may weight and update edge features.

## II. Applications of Graph Neural Networks

Graph neural networks are particularly suited to drug molecular graphs and protein structure graphs, effectively capturing complex structural information and higher-order relationships. Drugs consist of atoms (nodes) and chemical bonds (edges), while proteins can be represented by amino acid residues (nodes) and their interactions (edges). GNNs can be combined with other networks to predict interactions between drugs and molecules. For example, a GNN first learns molecular features, and fully connected layers then predict possible drug structures.

GNNs and models combining Transformers and convolutional neural networks (such as TC-DTA) also perform well in DTA prediction. They learn nonlinear mappings from complex drug and target features to predict affinity values. For example, the G-K BertDTA framework combines graph representation learning and semantic embeddings for drug–target affinity prediction.

## References

- Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2009). [The Graph Neural Network Model](https://doi.org/10.1109/TNN.2008.2005605). IEEE Transactions on Neural Networks.
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). [Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a.html). ICML 2017.
- Tang, X., Zhou, Y., Yang, M., & Li, W. (2024). [TC-DTA: Predicting Drug-Target Binding Affinity With Transformer and Convolutional Neural Networks](https://doi.org/10.1109/TNB.2024.3441590). *IEEE Transactions on NanoBioscience*, 23(4), 572–578.
- Qiu, X., Wang, H., Tan, X., & Fang, Z. (2024). [G-K BertDTA: A Graph Representation Learning and Semantic Embedding-Based Framework for Drug-Target Affinity Prediction](https://doi.org/10.1016/j.compbiomed.2024.108376). *Computers in Biology and Medicine*, 173, 108376.
